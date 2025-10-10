from NeuralNetwork import NN
import torch
from csvreader import read_csv

def main():
    print("Test")
    nn = NN(784,10)
    df = read_csv("train.csv")
    pred = nn.forward(df.loc[1,df.columns != "label"].values.tolist())
    print(pred)
    print(torch.argmax(pred).item())

if __name__ == "__main__":
    main()

