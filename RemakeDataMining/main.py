from NeuralNetwork import NN,Agent
import torch
from csvreader import read_csv
from random import randint
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df = pd.read_csv('sncb_data_challenge.csv', delimiter=';', index_col=False)
    print(df.loc[1,df.columns=="events_sequence"].values[0].strip("[]").split(", "))
    all_events = []
    for x in range(len(df)):
        for event in df.loc[x,df.columns=="events_sequence"].values[0].strip("[]").split(", "):
            if event not in all_events:
                all_events.append(event)

    print(all_events)    

main()