import pickle
import torch

train_file = open("data/train.pt", "rb")
train_data = pickle.load(train_file)
i = 0
