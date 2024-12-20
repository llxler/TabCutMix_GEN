import numpy as np
import pandas as pd
import os
import sys
import random
import json
import argparse
from process_data.cutmix_intra import cutmix_tabular

TYPE_TRANSFORM = {
    'float', np.float32,
    'str', str,
    'int', int
}

INFO_PATH = 'data_condition/Info'

parser = argparse.ArgumentParser(description='process dataset')

# General configs
parser.add_argument('--dataname', type=str, default="shoppers", help='Name of dataset.')
args = parser.parse_args()


def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None):
    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping = {}

    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train=0, num_test=0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    seed = 82

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1

    ############## 测试小数量级数据集 ##############
    # seed = 5201314
    # data_seed = data_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # train_df = data_seed[:num_train]
    # test_df = data_seed[-num_test:]
    ############## 测试小数量级数据集 ##############

    return train_df, test_df, seed


def process_data(name):
    with open(f'{INFO_PATH}/{name}.json', 'r') as f:

        info = json.load(f)

    data_path = info['data_path']
    print(data_path)
    if info['file_type'] == 'csv':
        data_df = pd.read_csv(data_path, header=info['header'])

    elif info['file_type'] == 'xls':
        data_df = pd.read_excel(data_path, sheet_name='Data', header=1)
        data_df = data_df.drop('ID', axis=1)

    # 取出数据的百分比
    # data_df = data_df.sample(frac=0.005, random_state=5201314)

    num_data = data_df.shape[0]

    column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx,
                                                                                 target_col_idx, column_names)
    print('num_col_idx', num_col_idx)
    print('cat_col_idx', cat_col_idx)
    print('column_names', column_names)
    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]
    print(num_columns)
    print(cat_columns)
    # print(target_columns)

    if info['test_path']:

        # if testing data is given
        test_path = info['test_path']

        with open(test_path, 'r') as f:
            lines = f.readlines()[1:]
            test_save_path = f'data_condition/{name}/test.data'
            if not os.path.exists(test_save_path):
                with open(test_save_path, 'a') as f1:
                    for line in lines:
                        save_line = line.strip('\n').strip('.')
                        f1.write(f'{save_line}\n')

        test_df = pd.read_csv(test_save_path)
        train_df = data_df

    else:

        # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)

        num_train = int(num_data * 1)
        num_test = 0

        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
        # print(data_df)
        # print(train_df)

    """ cutmix """
    print(data_df.columns)
    print(train_df.shape)
    print(type(train_df.shape[0]))
    num_new_samples = int(train_df.shape[0] * 0.3)
    print(name)
    if name == 'adult':
        label_idx = 14
    elif name == 'default':
        label_idx = 23
    elif name == 'shoppers':
        label_idx = 17
    elif name == 'magic':
        label_idx = 10
        num_new_samples = int(train_df.shape[0] * 2.0)
    if INFO_PATH == 'data_condition/Info_0.7':
        label_idx = 11
    # 注释掉下面这行可以不使用这个cutmix
    # train_df = cutmix_tabular(train_df, label_idx, num_new_samples)
    print(train_df.shape)

    """ use part of training """
    # train_percent = 0.5
    # train_df = train_df.sample(frac=train_percent, random_state=50)
    # train_percent = 0.6  # 30% data
    # train_df = train_df.sample(frac=train_percent, random_state=50)
    # train_percent = 0.3333333  # 10% data
    # train_df = train_df.sample(frac=train_percent, random_state=25)
    print('train_df.shape', train_df.shape)
    print('test_df.shape', test_df.shape)

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    print(name, train_df.shape, test_df.shape, data_df.shape)
    print(train_df)
    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        col_info['max'] = float(train_df[col_idx].max())
        col_info['min'] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        col_info['categorizes'] = list(set(train_df[col_idx]))
        print(col_info['categorizes'])
        print(col_idx)

    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info['type'] = 'numerical'
            col_info['max'] = float(train_df[col_idx].max())
            col_info['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info['type'] = 'categorical'
            col_info['categorizes'] = list(set(train_df[col_idx]))
            print(col_info['categorizes'])
            print(col_idx)

    info['column_info'] = col_info

    train_df.rename(columns=idx_name_mapping, inplace=True)
    test_df.rename(columns=idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'
    for col in num_columns:
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == '?', col] = 'nan'

    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()

    print(test_df[num_columns])
    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

    save_dir = f'data_condition/{name}'
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
    np.save(f'{save_dir}/y_train.npy', y_train)

    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
    np.save(f'{save_dir}/y_test.npy', y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    train_df.to_csv(f'{save_dir}/train.csv', index=False)
    test_df.to_csv(f'{save_dir}/test.csv', index=False)

    if not os.path.exists(f'synthetic/{name}'):
        os.makedirs(f'synthetic/{name}')

    train_df.to_csv(f'synthetic/{name}/real.csv', index=False)
    test_df.to_csv(f'synthetic/{name}/test.csv', index=False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping

    metadata = {'columns': {}}
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    for i in num_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in cat_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'

    if task_type == 'regression':

        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'

    else:
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'

    info['metadata'] = metadata

    with open(f'{save_dir}/info.json', 'w') as file:
        json.dump(info, file, indent=4)

    print(f'Processing and Saving {name} Successfully!')

    print(name)
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'])
    print('Test', info['test_num'])
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Cat', cat)


if __name__ == "__main__":

    if args.dataname:
        process_data(args.dataname)
    else:
        for name in ['adult', 'default', 'shoppers', 'magic']:
            process_data(name)



