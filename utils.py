import argparse
import matplotlib.pyplot as plt
import seaborn
import numpy as np


def parse_args():
    desc = 'PyTorch example code for Kaggle competition -- Plant Seedlings Classification.\n' \
           'See https://www.kaggle.com/c/plant-seedlings-classification'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-p', '--path', help='path to dataset', type=str, default="./data/plant-seedlings-classification-cs429529/")
    parser.add_argument('--pretrained_model_path', help='path to pretrained model', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=30)                   # number of epochs, default: 50
    parser.add_argument('--batch_size', type=int, default=32)               # batch size, default: 32
    parser.add_argument('--lr', type=float, default=0.001)                  # learning rate, default:0.001, 
    parser.add_argument('--num_workers', type=int, default=4)               # number of workers, default: 4
    parser.add_argument('--cuda', action='store_true', default=True)    # disable cuda device, default: False
    return parser.parse_args()

def confusion_matrix(calc_class, real_class):

    conf_matrix = np.zeros((12, 12), dtype=np.int32)

    for row in range(12):
        for col in range(12):
            conf_matrix[row][col] = np.sum(calc_class[real_class == col] == row)
    
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Confusion Matrix")

    seaborn.heatmap(conf_matrix, annot=True, fmt=".0f")#, xticklabels=news_groups, yticklabels=news_groups) # plot
    plt.show()