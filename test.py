from TrainCSV.train import Train_From_CSV
from utils.csvnorm.csvnorm import process_csv

# process_csv(input_file='Data/train_10000.csv', output_file='Data/norm_train_10000.csv')
Train_From_CSV(in_path='Data/norm_train_10000.csv', out_path='weights/', sock='', echo=False)