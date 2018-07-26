#!/usr/bin/env python
from shutil import copy
import subprocess
def split():
    root = '/home/interns/xuan/datasets/SUNRGBD/'
    label_src = '/home/interns/xuan/datasets/sunrgbd_train_test_labels/'
    train_src = '/home/interns/xuan/datasets/SUNRGBD-train_images/'
    test_src = '/home/interns/xuan/datasets/SUNRGBD-test_images/'

    # clear folders
    print('Clearing folders...')
    subprocess.Popen('rm '+root + 'test/*', shell=True).wait()
    subprocess.Popen('rm '+root + 'train/*', shell=True).wait()
    subprocess.Popen('rm '+root + 'annotations/test/*', shell=True).wait()
    subprocess.Popen('rm '+root + 'annotations/train/*', shell=True).wait()

    #test
    size_train = 5285#5285
    size_test = 5050#5050
    for i in range(1, 1+size_test):
        src = test_src+ 'img-%06d.jpg' % i
        dst = root + 'test'
        copy(src, dst)
        print('copying ' + src + ' to '+ dst)
    #train
    for i in range(1, 1+size_train):
        src = train_src+ 'img-%06d.jpg' % i
        dst = root + 'train'
        copy(src, dst)
        print('copying ' + src + ' to '+ dst)

    # val test
    for i in range(1, 1+size_test): #range(1, 5051)
        src = label_src+ 'img-%06d.png' % i
        dst = root+'annotations/test'
        copy(src, dst)
        print('copying ' + src + ' to '+ dst)
    # val train
    for i in range(5051, 5051+size_train): #range(5051, 10336)
        src = label_src+ 'img-%06d.png' % i
        dst = root+'annotations/train'
        copy(src, dst)
        print('copying ' + src + ' to '+ dst)

if __name__ == '__main__':
    split()
