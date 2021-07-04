import pandas as pd
import datatable as dt
import numpy as np
from sklearn import preprocessing
from category_encoders.leave_one_out import LeaveOneOutEncoder
from pandas import pandas


# 数据可视化
import seaborn as sns
import missingno as msno
import matplotlib.pyplot  as plt

# 一键数据分析
import pandas_profiling

# multi-hot 
from sklearn.preprocessing import MultiLabelBinarizer

# lda降维
from sklearn.decomposition import LatentDirichletAllocation as LDA

import sys
import logging
import os
import gensim
# 引入doc2vec
from gensim.models import Doc2Vec
import re
# from utilties import ko_title2words

import jieba





#加载用户数据为dataFrame
def load_user():
    user_data = dt.fread("../../dataset/traindata/user_features_data/user_features_data.csv")
    # 将user_data从datatable转为dataframe
    user_df = user_data.to_pandas()
    return user_df

def load_user_modified():
    user_data = dt.fread("../../dataset/traindata/user_features_data/user_data_v2.csv")
    # 将user_data从datatable转为dataframe
    user_df = user_data.to_pandas()
    return user_df

# 加载视频数据为dataFrame
def load_video():
    # 读取为datatable
    video_data = dt.fread("../../dataset/traindata/video_features_data/video_features_data.csv")
    # 转换为dataframe
    video_df = video_data.to_pandas()
    return video_df

# 加载视频数据为dataFrame
def load_video_modifiled():
    # 读取为datatable
    video_data = dt.fread("../../dataset/traindata/video_features_data/video_data_v3.csv")
    # 转换为dataframe
    video_df = video_data.to_pandas()
    return video_df



def load_actions(all_features=False):
    if all_features:
        actions_data = dt.fread("../../dataset/traindata/history_behavior_data/all_action_features.csv")
        actions_df = actions_data.to_pandas()
        return actions_df
    else:
        actions_data = dt.fread("../../dataset/traindata/history_behavior_data/all_action_data.csv")
        actions_df = actions_data.to_pandas()
        return actions_df
        


def load_csv(pth):
    df = dt.fread(pth).to_pandas()
    return df
# 保存用户数据
def save_user_data(df, name):
    return df.to_csv(path_or_buf=f"../../dataset/traindata/user_features_data/{name}.csv", columns=df.columns.values.tolist(), index=None)

# 保存视频数据
def save_video_temp_data(df, name):
    df.to_csv(path_or_buf=f"../../dataset/traindata/video_features_data/{name}.csv", columns=df.columns.values.tolist(), index=None)
    print("保存成功\n保存路径为：",f"../../dataset/traindata/video_features_data/{name}.csv")

# 获取某一列特征的multi-hot变量
def getMultiHotLabel(df, col_name, sep=','):
    mlb = MultiLabelBinarizer()
    tmp = df[col_name][:].apply(lambda x : x.split(sep))
    return mlb.fit_transform(tmp)

# 将高维编码降维为低维编码dddd
def getLDAencode(encode_list, n_topics = 16):
    lda = LDA(n_components=n_topics, max_iter=5,
              learning_method='online',
              learning_offset=50.,
              random_state=0)
    return pd.DataFrame(lda.fit_transform(encode_list), index=None)


# 拼接两表，并删除多余属性列。
def concatDf(source_df, new_df, drop_col_name):
#     new_df = pd.DataFrame(new_df, index=None)
    
    col_name = drop_col_name
    new_col_names = new_df.columns.values
    new_df.columns = [f"{col_name}_{col}" for col in new_col_names] 
    
    source_df = pd.concat([source_df, new_df], axis=1)
    source_df.drop([drop_col_name], axis=1, inplace=True)
    print("拼接成功")
#     source_df.info() 
    return source_df


def renameCol(df, old_col_name, new_col_name):
    df = df.rename(columns = {df.columns[-1]:new_col_name})
    return df


def couerGroupToDF(groups, fun, col_name):
    df = pd.DataFrame(groups)
    df = renameCol(df, fun, col_name)
    return df