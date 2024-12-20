import pandas as pd
import numpy as np

column_indices = {
    'magic': {
        'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'categorical': [10]
    },
    'adult': {
        'numerical': [0, 2, 4, 10, 11, 12],
        'categorical': [1, 3, 5, 6, 7, 8, 9, 13, 14]
    },
    'default': {
        'numerical': [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        'categorical': [1, 2, 3, 5, 6, 7, 8, 9, 10, 23]
    },
    'Churn_Modelling': {
        'numerical': [0,3,4,5,6,9],
        'categorical': [1,2,7,8,10]
    },
    'cardio_train': {
        'numerical': [0,2,3,4,5],
        'categorical': [1,6,7,8,9,10,11]
    },
    'wilt': {
        'numerical': [1,2,3,4,5],
        'categorical': [0]
    },
    'MiniBooNE': {
        'numerical': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        'categorical': [0]
    },
    'shoppers': {
        'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'categorical': [10, 11, 12, 13, 14, 15, 16, 17]
    },
}

# 假设你的 DataFrame 叫做 df
df = pd.read_csv('E:\TabCutMix\data\shoppers\\online_shoppers_intention.csv')  # 如果是从文件读取数据

# 可控的概率 p
p = 0.5  # 你可以调整这个值

# 假设 column_indices['shoppers']['categorical'] 已经定义
categorical_columns = column_indices['shoppers']['categorical']

# 获取需要修改的行的位置
change_indices = np.random.choice(df.index, size=int(len(df) * p), replace=False)

# 遍历选中的行，逐个将 categorical 列的值改为 "MASK"
for idx in change_indices:
    for col_idx in categorical_columns:
        df.iat[idx, col_idx] = "MASK"

# 保存修改后的 DataFrame 到文件
df.to_csv('E:\TabCutMix\data\shoppers\\online_shoppers_intention_condition.csv', index=False)  # 修改文件名为你想保存的文件名

# 打印确认修改
print("数据已保存到 'modified_data.csv'")
