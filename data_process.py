import pandas as pd
import matplotlib.pyplot as plt

csv_file = './logs/2801.csv'

def plt_csv(file):
    df = pd.read_csv(file)

    y2 = df['train loss'].values
    y3 = df['true loss'].values
    y1 = df['val acc'].values

    epoch_num = len(y1)
    x1 = range(0, epoch_num)
    x2 = range(0, epoch_num)
    x3 = range(0, epoch_num)

    plt.subplot(3, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.ylabel('Valid Accuracy')
    plt.subplot(3, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.subplot(3, 1, 3)
    plt.plot(x3, y3, '.-')
    plt.xlabel('epochs')
    plt.ylabel('True Loss')
    plt.savefig('./logs/2801.png')

plt_csv(csv_file)