from NeuralNetwork import NN,Agent
import torch
from csvreader import read_csv
from random import randint


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

def main():
    df = read_csv("train.csv")
    accuracies = []
    for fold in range(10):
        agent=Agent(784,10)
        accuracy=0
        train, test = get_lines(10,fold,42000)
        for j in train:
            print(j)
            target = [0] * 10
            x = j #randint(0,28000)
            target[df.loc[x,df.columns == "label"].values[0]]=1000
            
            agent.train(df.loc[x,df.columns != "label"].values.tolist(),target)

        for j in test:
            print(j)
            if(torch.argmax(agent.forward(df.loc[j,df.columns != "label"].values.tolist())).item()== df.loc[j,df.columns == "label"].values[0]):
                accuracy+=1
        accuracies.append(accuracy/(42000/10))
    print(sum(accuracies)/len(accuracies))
    print(len(df)) # 42000


if __name__ == "__main__":
    main()

