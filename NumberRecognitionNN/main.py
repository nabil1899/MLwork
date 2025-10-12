from NeuralNetwork import NN,Agent
import torch
from csvreader import read_csv

def main():
    df = read_csv("train.csv")
    agent=Agent(784,10)

    accuracy=0
    for j in range(500):
        target = [0] * 10
        target[df.loc[j,df.columns == "label"].values[0]]=1000

        for i in range(500):
            agent.train(df.loc[j,df.columns != "label"].values.tolist(),target)
        if(torch.argmax(agent.forward(df.loc[j,df.columns != "label"].values.tolist())).item()== df.loc[j,df.columns == "label"].values[0]):
            accuracy+=1

    print(accuracy/500)


if __name__ == "__main__":
    main()

