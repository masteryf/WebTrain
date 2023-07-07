import torch
import pandas as pd
from torch import optim
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
from utils.MyDataset_FromCSV_Data import MyDataset
from Models.Model import *
from TrainCSV.trainUtils import test_line
from TrainCSV.trainUtils import train_line
import time
import datetime


def Train_From_CSV(path, BATCH_SIZE=32, EPOCHS=100, DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   LEARNRATE=0.00001):
    df: DataFrame = pd.read_csv(path, encoding='utf-8')
    df.fillna(df.mean(), inplace=True)  # 处理缺失值，用每一列的平均值填充

    # 标准化数据
    # features = [f'feature{i}' for i in range(106)]
    # df[features] = (df[features] - df[features].mean()) / df[features].std()

    df = df.sample(frac=1)
    cut_idx = int(round(0.1 * df.shape[0]))
    df_test, df_train = df.iloc[:cut_idx].reset_index(drop=True), df.iloc[cut_idx:].reset_index(drop=True)

    print(df_test)

    Train = MyDataset(df_train)
    Test = MyDataset(df_test)

    train_loader = torch.utils.data.DataLoader(Train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Test, shuffle=True)

    model = NN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNRATE)

    timestamp = time.time()
    datetime_obj = datetime.datetime.fromtimestamp(timestamp)
    starttime = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')

    for epoch in range(1, EPOCHS + 1):
        LEARNRATE *= 0.95
        print(LEARNRATE)
        train_line.train(model, DEVICE, train_loader, optimizer, epoch)
        test_line.test(model, DEVICE, test_loader)
        torch.save(NN.state_dict(), "weights/" + starttime + ".pth")

    return "weights/" + starttime + ".pth"