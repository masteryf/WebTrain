import torch
import pandas as pd
from torch import optim
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
from utils.MyDataset_FromCSV_Data import MyDataset
from Models.Model import *
from utils import test_line
from utils import train_line
import time
import datetime
from utils.MyDataset_FromCSV_Data import MyDataset_NoLabel

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN().to(DEVICE)

def prediction(path):
    df: DataFrame = pd.read_csv(path, encoding='utf-8')
    data = MyDataset_NoLabel(df)
    output = model(data)
    return output