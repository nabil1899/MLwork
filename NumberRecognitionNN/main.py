from NeuralNetwork import NN,Agent
import torch
from csvreader import read_csv
from random import randint
import matplotlib.pyplot as plt

NB_FOLDS=5
def get_lines(k,index,number):
    train = []
    test = []
    fold_size = number / k
    for i in range(number):
        if i >= fold_size* index and i < fold_size* (index+1):
            test.append(i)
        else:
            train.append(i)
    
    return train, test
def normalize(list):
    result=[]
    for i in list:
        result.append(i/255)
    return result
def hiddenLayerComparaison(hidden):
    df = read_csv("train.csv")
    accuracies = []

    for fold in range(NB_FOLDS):
        agent=Agent(784,10,hidden)
        accuracy=0
        train, test = get_lines(NB_FOLDS, fold, 42000)
        for j in train:
            x = j  # randint(0,28000)
            """target = [0] * 10
            
            target[df.loc[x,df.columns == "label"].values[0]]=1000"""
            target = int(df.loc[x,df.columns == "label"].values[0])

            agent.train(normalize(df.loc[x,df.columns != "label"].values.tolist()),target)

        for j in test:

            if(torch.argmax(agent.forward(normalize(df.loc[j,df.columns != "label"].values.tolist()))).item()== df.loc[j,df.columns == "label"].values[0]):
                accuracy+=1
        accuracies.append(accuracy / (42000 / NB_FOLDS))

    return sum(accuracies)/len(accuracies)


if __name__ == "__main__":
    accuracies=[]
    for i in range(80,96,16):
        print(i)
        accuracies.append(hiddenLayerComparaison(i))
    print(accuracies)
    plt.plot(accuracies)
    plt.show()



