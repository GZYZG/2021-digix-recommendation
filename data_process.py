import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd
import datatable as dt
import time
from collections import Counter
# 归一化
from sklearn.preprocessing import QuantileTransformer


SEED = 0


# 节约内存的一个标配函数
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem, 100*(start_mem-end_mem)/start_mem, (time.time()-starttime)/60))
    return df

def load_train_datas(path = "../完整版df_train.jay"):
    df_train = reduce_mem(dt.fread(path).to_pandas())
    # 去除 watch_label = 0的点
    df_train = df_train[df_train['is_watch'] == 1]
    # 删除 video_name、is_watch、video_id、user_id 列
    df_train.drop(['video_name','is_watch','date', 'user_id', 'video_id'], axis=1, inplace=True)
    
    dataset = df_train
    # 准备数据
    watch_label = dataset.pop('watch_label').astype(np.uint8)
    is_share = dataset.pop('is_share').astype(np.uint8)
    
    return dataset, watch_label, is_share
    

def load_test_datas(path="../完整版df_test.jay"):
    # 拼接好的测试数据集
    df_test = reduce_mem(dt.fread(path).to_pandas())
    # 删除 video_name、 video_id、user_id、date 列
    df_test.drop(['video_name','user_id', 'video_id','date'], axis=1, inplace=True)
    # 填充缺失值
    df_test.fillna(value=0, inplace=True)
    return df_test


def split_dataset(train_dataset, wt_label, sh_label, val_ratio=0.15):    
    # 将train划分为 train、validation. validation占ratio%。
    validation_indices = train_dataset.sample(frac=val_ratio, replace=False, random_state=SEED).index
    validation_data = train_dataset.iloc[validation_indices]
    validation_label = [wt_label[validation_indices], sh_label[validation_indices]]

    train_indices = list(set(train_dataset.index) - set(validation_indices))
    train_data = train_dataset.iloc[train_indices]
    train_label = [wt_label[train_indices], sh_label[train_indices]]
    
    return train_data, train_label, validation_data, validation_label

def getTensorDataset(my_x, my_y):
    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(my_y)
    return torch.utils.data.TensorDataset(tensor_x, tensor_y)


def getScalarData(df):
    return QuantileTransformer(output_distribution='uniform').fit_transform(df)

def reSample_WatchLabels(dataset, watch_label, is_share):
    items = list(Counter(watch_label).items())
    
    under_ss = np.array(items)
    under_ss_thresh = under_ss[3, 1]  # 设置每个类别样本数目的上限 [219188], 超过上限按上限计算
    under_ss[:, 1] = np.clip(under_ss[:, 1], a_min=None, a_max=under_ss_thresh)

    over_ss = under_ss.copy() 
    over_ss_thresh = under_ss[2, 1]  # 设置每个类别样本数据的下限，此时under_ss为 219188, 低于下限按下限计算。
    over_ss[:, 1] = np.clip(over_ss[:, 1], a_min=over_ss_thresh, a_max=None)

    under_ss = dict(under_ss)
    over_ss = dict(over_ss)
    
    idxs = watch_label == 0
    idxs = idxs.replace(False, np.nan).dropna().index  # 保留watch_label=0的行索引
    
    # 随机抽样
    left_idxs = np.random.choice(idxs, under_ss_thresh, replace=False)  # 选择一部分保留，注意replace参数，为True时会重复采样
    del_idxs = idxs.difference(left_idxs)
    del_idxs.shape, left_idxs.shape
    
    resampled_data = np.delete(dataset.values, del_idxs, axis=0)
    resampled_wl = np.delete(watch_label.values, del_idxs, axis=0)
    resampled_sl = np.delete(is_share.values, del_idxs, axis=0)
    
    # 将采样后的数据重装回 DataFrame
    resampled_dataset = pd.DataFrame(resampled_data, columns=dataset.columns)
    watch_label_res = pd.Series(resampled_wl)
    share_label_res = pd.Series(resampled_sl)
    
    return resampled_data, watch_label_res, share_label_res
    
    
def to_categorical(y, num_classes=None, dtype='float32'):
    """
    From keras sorucecode: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        torch.save(model, 'finish_model.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss