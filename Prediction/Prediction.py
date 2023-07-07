import pandas as pd
from pandas.core.frame import DataFrame
from Models.Model import *
from utils.dataset.MyDataset_FromCSV_Data import MyDataset_NoLabel

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN().to(DEVICE)


def prediction(path,model_path = ""):
    df: DataFrame = pd.read_csv(path, encoding='utf-8')
    data = MyDataset_NoLabel(df)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    output = model(data)
    return output