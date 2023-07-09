import pandas as pd
import numpy as np

def process_csv(input_file, output_file):
    # 读取csv文件
    df = pd.read_csv(input_file)

    # 创建一个空的DataFrame用于存储平均值和A值
    mean_df = pd.DataFrame(columns=['feature', 'mean', 'A'])

    # 遍历特征，计算平均值并替换空值，然后除以A值
    for i in range(107):
        feature_name = f'feature{i}'
        # 计算平均值
        mean_value = df[feature_name].replace('nan', np.nan).astype(float).mean()
        # 用平均值填充空值
        df[feature_name] = df[feature_name].replace('nan', np.nan).astype(float).fillna(mean_value)
        # 减去平均值
        df[feature_name] = df[feature_name] - mean_value
        # 计算A值 (标准差)
        A = df[feature_name].std()
        # 如果标准差为0，则将其设置为1，避免除以0的错误
        if A == 0:
            A = 1
        # 除以A值
        df[feature_name] = df[feature_name] / A
        # 将平均值和A值添加到mean_df中
        new_row = pd.DataFrame({'feature': [feature_name], 'mean': [mean_value], 'A': [A]})
        mean_df = pd.concat([mean_df, new_row], ignore_index=True)

    # 将处理后的DataFrame输出到新csv文件
    df.to_csv(output_file, index=False)
    # 将平均值和A值DataFrame输出到新csv文件

