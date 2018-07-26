import csv
import matplotlib.pyplot as plt

if __name__=='__main__':
    with open('../training/2018-06-19/training_loss.csv', 'r') as loss:
        loss_reader = csv.reader(loss, delimiter = ' ')
        num_epoch = 59
        train_loss = [0.0]*num_epoch
        count = [0]*num_epoch
        for row in loss_reader:
            epoch = int(row[0])
            loss = float(row[1])
            train_loss[epoch-1]+= loss
            count[epoch-1]+=1
        for i in range(num_epoch):
            train_loss[i]/=count[i]
        print(range(1, num_epoch+1), train_loss)
        plt.plot(train_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
