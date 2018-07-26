from __future__ import division
from __future__ import print_function
import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
from ptsemseg.utils import convert_state_dict
import matplotlib.pyplot as plt
import json
from time import time

# Freeze batch norm 2d layers
# Use model.applay(freeze_batchnorm2d) to apply to all children
def freeze_batchnorm2d(m):
    if m.__class__.__name__ == 'BatchNorm2d':
        m.eval()
        # Don't update
        for param in m.parameters():
            param.requires_grad = False


def train(args):
    do_finetuning = True
    data_parallel = False # whether split a batch on multiple GPUs to accelerate
    if data_parallel:
        print('Using data parallel.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # use GPU1 if train on one GPU
    # Setup Augmentations
    data_aug= Compose([RandomSized(args.img_rows),
                        RandomHorizontallyFlip(),
                        RandomSizedCrop(args.img_rows)])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=False)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), img_norm=False)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last = True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8, drop_last = True)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()
        # window for training loss
        loss_window = vis.line(X=torch.ones((1)),
                           Y=torch.zeros((1)),
                           opts=dict(xlabel='epoch',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss'],
                                     width = 400,
                                     height = 400))
        # window for example training image
        image_train_window = vis.images(torch.zeros((3, 3, args.img_rows, args.img_cols)),
                           opts=dict(nrow = 3,
                                     caption = 'input-prediction-groundtruth',
                                     title = 'Training example image'))
        # window for train and validation accuracy
        acc_window = vis.line(X=torch.ones((1,2)),
                           Y=torch.zeros((1,2)),
                           opts=dict(xlabel='epoch',
                                     ylabel='mean IoU',
                                     title='Mean IoU',
                                     legend=['train','validation'],
                                     width = 400,
                                     height = 400))

        # window for example validation image
        image_val_window = vis.images(torch.zeros((3, 3, args.img_rows, args.img_cols)),
                           opts=dict(nrow = 3,
                                     caption = 'input-prediction-groundtruth',
                                     title = 'Validation example image'))
    # Setup Model
    model_name = 'pspnet'
    model = get_model(model_name, n_classes, version = args.dataset+'_res50')
    #model = get_model(model_name, n_classes, version = args.dataset+'_res101')
    if do_finetuning:
        # pspnet pretrained on ade20k
        pretrained_model_path = '/home/interns/xuan/models/pspnet_50_ade20k.pth'
        # pspnet pretrained on pascal VOC
        #pretrained_model_path = '/home/interns/xuan/models/pspnet_101_pascalvoc.pth'
        pretrained_state = convert_state_dict(torch.load(pretrained_model_path)['model_state']) # remove 'module' in keys
        # Load parameters except for last classification layer to fine tuning
        print('Setting up for fine tuning')
        # 1. filter out unnecessary keys
        pretrained_state = {k: v for k, v in pretrained_state.items() if k not in ['classification.weight', 'classification.bias',
                                                                                    'aux_cls.weight', 'aux_cls.bias']}
        # 2. overwrite entries in the existing state dict
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_state)
        # 3. load the new state dict
        model.load_state_dict(model_state_dict)

    # load checkpoint to continue training
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            print("Loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    '''
    # freeze all parameters except for final classification if doing fine tuning
    if do_finetuning:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classification.parameters():
            param.requires_grad = True
        for param in model.cbr_final.parameters():
            param.requires_grad = True
    '''

    # Set up optimizer
    opt_dict = {'name': 'SGD', 'learning_rate': args.l_rate, 'momentum': 0.9, 'weight_decay': 1e-3}
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opt_dict['learning_rate'], opt_dict['momentum'], opt_dict['weight_decay'])
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.9**epoch)

    # train on multiple GPU
    if data_parallel:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # move parameters to GPU
    model = model.to(device)

    best_iou = -100.0
    statistics = {}
    best_model_stat = {}
    # print params
    print('optimizer', opt_dict)
    print('batch size', args.batch_size)
    since = time() # start time
    # for every epoch.train then validate. Keep the best model in validation
    for epoch in range(1, args.n_epoch + 1):
        print('=>Epoch %d / %d' % (epoch, args.n_epoch))
        # -------- train --------
        model.train()
        # Freeze BatchNorm2d layers because we have small batch size
        #print('Freeze BatchNorm2d layers')
        #model.apply(freeze_batchnorm2d)
        print('  =>Training')
        loss_epoch = 0. # average loss in an epoch
        #scheduler.step()
        for i, (images, labels) in tqdm(enumerate(trainloader), total = len(trainloader)):
            images = images.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(images)
            # if use aux loss, loss_fn = multi_scale_cross_entropy2d
            # if ignore aux loss, loss_fn = cross_entropy2d
            loss = multi_scale_cross_entropy2d(input=outputs, target=labels, device = device)
            loss.backward()
            optimizer.step()
            # update average loss
            loss_epoch += loss.item()
            # update train accuracy (mIoU)
            pred = outputs[0].data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            running_metrics.update(gt, pred)

        loss_epoch /= len(trainloader)
        print('Average training loss: %f' % loss_epoch)
        # draw train loss every epoch
        if args.visdom:
            vis.line(
                X=torch.Tensor([epoch]),
                Y=torch.Tensor([loss_epoch]).unsqueeze(0),
                win=loss_window,
                update='append')
        # get train accuracy for this epoch
        scores_train, class_iou_train = running_metrics.get_scores()
        running_metrics.reset()
        print('Training mean IoU: %f' % scores_train['Mean IoU'])

        # -------- validate --------
        model.eval()
        print('  =>Validation')
        with torch.no_grad():
            for i_val, (images_val, labels_val) in tqdm(enumerate(valloader), total = len(valloader)):
                images_val = images_val.to(device)
                labels_val = labels_val.to(device)
                outputs = model(images_val)
                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()
                running_metrics.update(gt, pred)

        scores_val, class_iou_val = running_metrics.get_scores()
        running_metrics.reset()
        for k, v in scores_val.items():
            print(k+': %f' % v)

        # --------save best model --------
        if scores_val['Mean IoU'] >= best_iou:
            best_iou = scores_val['Mean IoU']
            best_model = model.state_dict()
            if data_parallel:
                best_model = convert_state_dict(best_model) # remove 'module' in keys to be competible with single GPU
            torch.save(best_model, "{}_{}_best_model.pth".format(model_name, args.dataset))
            print('Best model updated!')
            print(class_iou_val)
            best_model_stat = {'epoch': epoch, 'scores_val': scores_val, 'class_iou_val': class_iou_val}

        # -------- draw --------
        if args.visdom:
            # draw accuracy for training and validation
            vis.line(
                X=torch.Tensor([epoch]),
                Y=torch.Tensor([scores_train['Mean IoU'], scores_val['Mean IoU']]).unsqueeze(0),
                win=acc_window,
                update='append')
            # show example train image
            with torch.no_grad():
                (image_train, label_train) = t_loader[0]
                gt = t_loader.decode_segmap(label_train.numpy())
                image_train = image_train.unsqueeze(0)
                image_train = image_train.to(device)
                label_train = label_train.to(device)
                outputs = model(image_train)
                pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
                decoded = t_loader.decode_segmap(pred)
                vis.images([image_train.data.cpu().squeeze(0), decoded.transpose(2,0,1)*255.0, gt.transpose(2,0,1)*255.0], win = image_train_window)
	        # show example validation image
            with torch.no_grad():
                (image_val, label_val) = v_loader[0]
                gt = v_loader.decode_segmap(label_val.numpy())
                image_val = image_val.unsqueeze(0)
                image_val = image_val.to(device)
                label_val = label_val.to(device)
                outputs = model(image_val)
                pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
                decoded = v_loader.decode_segmap(pred)
                vis.images([image_val.data.cpu().squeeze(0), decoded.transpose(2,0,1)*255.0, gt.transpose(2,0,1)*255.0], win = image_val_window)

        # -------- save training statistics --------
        statistics['epoch %d' % epoch] = {'train_loss': loss_epoch, 'scores_train': scores_train, 'scores_val': scores_val}
        with open('train_statistics.json', 'w') as outfile:
            json.dump({
                    'optimizer': opt_dict,
                    'batch_size': args.batch_size,
                    'data_parallel': data_parallel,
                    'Training hours': (time() - since)/3600.0,
                    'best_model': best_model_stat,
                    'statistics': statistics
                    }, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--visdom', dest='visdom', action='store_true',
                        help='Enable visualization(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false',
                        help='Disable visualization(s) on visdom | False by default')
    parser.set_defaults(visdom=False)

    args = parser.parse_args()
    train(args)
