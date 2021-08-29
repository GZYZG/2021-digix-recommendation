from copy import deepcopy
import pandas as pd
import numpy as np
from datatable import join
import datatable as dt
from sklearn import metrics
import os
from sklearn.model_selection import StratifiedKFold, KFold
from time import time
import xgboost as xgb


##################################################指标##################################################
def confusion_matrix(label, predict, n):
    """
    计算混淆矩阵
    :param label: 标签，np.array类型。形状可以是(n_sample,) 或者 (n_sample, n_classes)，当为第二种形状时可以表示多标签分类的情况
    :param predict: 预测值，与 `label` 同理
    :param n: 类别数目
    :return: 混淆矩阵，np.array类型。shape 为 (n, n)。$cm_{ij}$表示真实标签为 $i$，预测标签为 $j$ 的样本个数
    """
    k = (label >= 0) & (label < n)
    # bincount()函数用于统计数组内每个非负整数的个数
    # 详见 https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html
    return np.bincount(n * label[k].astype(int) + predict[k], minlength=n ** 2).reshape(n, n)

def auc(y, p, classes):
    """
    给定真实标签和预测标签，计算每个类别的auc值。实际只算出了roc曲线上一个点，即一个(fpr, tpr)，再并上(0, 0)和(1, 1)来计算auc
    :param y: 标签，np.array类型
    :param p: 预测标签，np.array类型
    :param classes: 类别，list-like，表示有哪些类别
    
    ----------
    Examples: 
    y = np.random.randint(0, 10, 100)
    p = np.random.randint(0, 10, 100)
    
    classes = list(range(10))
    weights = np.arange(0, 1, 0.1)
    all_aucs = auc(y, p, classes)

    weighted_auc = (all_aucs * weights).sum()
    print(f"{all_aucs}\n{weighted_auc}")
    """
    all_aucs = np.zeros(len(classes))
    for i, c in enumerate(classes):
        _y = np.zeros_like(y)
        _y[y==c] = 1
        _y[y!=c] = 0
        _p = np.zeros_like(p)
        _p[p==c] = 1
        _p[p!=c] = 0
#         print(_y, _p)
        cm = confusion_matrix(_y, _p, 2)
#         print(cm)
        tpr = (cm[0, 0] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) != 0 else 0
        fpr = (cm[1, 0] / (cm[1, 0] + cm[1, 1])) if (cm[1, 0] + cm[1, 1]) != 0 else 0
        tpr = [0, tpr, 1]
        fpr = [0, fpr, 1]
        auc = metrics.auc(fpr, tpr)
        all_aucs[i] = auc
        if _y.sum() == 0 or _p.sum() == 0:
            all_aucs[i] = 0
    return all_aucs


##################################################数据IO##################################################
def load_npz(path):
    npz = np.load(path, allow_pickle=True)
    return npz

def load_table(path, ftype="csv", data_name="data", column_name="columns"):
    if ftype == "npz":
        npz = load_npz(path)
        tab = pd.DataFrame(npz[data_name], columns=column_name)
    elif ftype == "jay":
        tab = dt.fread(path)
    elif ftype == "csv":
        tab = pd.read_csv(path)
        
    return tab
        
def merge_user_video_action(user, video, action, return_others=False):
    """
    将用户特征矩阵、视频特征矩阵、行为拼接起来
    """
    tab_user = dt.fread(user) if isinstance(user, str) else dt.Frame(user)
    tab_video = dt.fread(video) if isinstance(video, str) else dt.Frame(video)
    tab_act = dt.fread(action) if isinstance(action, str) else dt.Frame(action)
    
    tab_user.key = 'user_id'
    tab_act_user = tab_act[:, :, join(tab_user)]
    tab_video.key = 'video_id'
    tab_act_user_video = tab_act_user[:, :, join(tab_video)]
    
    if not return_others:
        return tab_act_user_video 
    else:
        return tab_act_user_video, {"user": tab_user, "video": tab_video, "action": tab_act}

def load_train_test_data(path=None, pre_merged=True, return_others=False, **kwargs):
    """
    读取保存的训练数据
    """ 
    if pre_merged:
        assert path is not None
        tab = dt.fread(path)
#     del tab[:, ['video_id', 'user_id']]
        return tab
    else:
        p_user = kwargs.get('p_user')
        p_video = kwargs.get('p_video')
        p_action = kwargs.get('p_action')
        
        if return_others:
            tab, others = merge_user_video_action(p_user, p_video, p_action, return_others=True)
            return tab, others
        else:
            tab = merge_user_video_action(p_user, p_video, p_action)
            return tab

def read_npz_to_df(path, data_name='data', column_name='columns'):
    npz = np.load(path, allow_pickle=True)
    df = pd.DataFrame(npz[data_name], columns=npz[column_name])
    
    return df


##################################################调参##################################################
def gridsearch_xgb(grid_params, xg_train, xg_test, n_round=200, n_class=10, verbose_eval=True):
    """
    xg_train, xg_test: DMatrix
    """
    results = []
    n_param = len(grid_params)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    for i, p in enumerate(grid_params):
        t0 = time()
        model = xgb.train(p, xg_train, n_round, watchlist, verbose_eval=verbose_eval)
        print(f"{n_round}-rounds Training finished ...\t\t({time()-t0:.3f}s)")

#         # get prediction
        pred = model.predict(xg_test)
#         # pred = pred.astype(np.uint8)
#         labels = xg_test.get_label()
#         error_rate = np.sum(pred != labels) / labels.shape[0]
#         print('Test error using softmax = {}'.format(error_rate))
        
        # eval the test using model
        evals = model.eval(xg_test)
        eval_dict = eval_str_2_dict(evals)

        aucs = auc(labels.astype(np.uint8), pred.astype(np.uint8), np.arange(n_class))
        if n_class == 2:
            w_aucs = 0
        else:
            weights = np.arange(0, 1, 0.1)
            w_aucs = (aucs * weights).sum()

        rep = metrics.classification_report(list(labels), list(pred))
        rep_df = report_2_df(rep)
        results.append({
            'param': p,
            'aucs': aucs,
            'w_auc': w_aucs,
            'report': rep_df,
            'model': model
        })
        results[-1].update(eval_dict)

        print(f"{i+1} / {n_param} : {n_round}-rounds Training finished ...\t\t({time()-t0:.3f}s)")
    
    return results

def cv_xgb(param, cv, X, y, n_round=200, n_class=10, n_splits=5, stratified=True, shuffle=True, random_state=444, verbose_eval=True)->dict:
    if isinstance(cv, int):
        if stratified:
            cv = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
        else:
            cv = KFold(n_splits, shuffle=shuffle, random_state=random_state)
    
    results = {}
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    i = 0
    n_splits = cv.n_splits
    for train_idxs, test_idxs in cv.split(X, y):
        X_train = X[train_idxs]
        X_test  = X[test_idxs]
        y_train = y[train_idxs]
        y_test  = y[test_idxs]
        
        xg_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        xg_test = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        
        
        t0 = time()
        model = xgb.train(param, xg_train, n_round, watchlist, verbose_eval=verbose_eval)
#         print(f"{n_round}-rounds Training finished ...\t\t({time()-t0:.3f}s)")

#         # get prediction
        pred = model.predict(xg_test)
#         # pred = pred.astype(np.uint8)
#         error_rate = np.sum(pred != y_test) / y_test.shape[0]
#         print('Test error using softmax = {}'.format(error_rate))
        
        # eval the test using model
        evals = model.eval(xg_test)
        eval_dict = eval_str_2_dict(evals)
        
        aucs = auc(y_test.astype(np.uint8), pred.astype(np.uint8), np.arange(n_class))
        # aucs[aucs == 0.5] = 0
        if n_class == 2:
            w_aucs = 0
        else:
            weights = np.arange(0, 1, 0.1)
            w_aucs = (aucs * weights).sum()


        rep = metrics.classification_report(list(y_test), list(pred), output_dict=True)
        rep_df = report_2_df(rep)
        
        # 处理每一fold的结果，对每个指标进行平均
        items = [(e[0], e[1] / n_splits) for e in eval_dict.items()]
        eval_dict = dict(items)
        if not results:
            results = {
                'aucs': aucs / n_splits,
                'w_auc': w_aucs / n_splits,
                'report': rep_df / n_splits,
    #             'model': model,
    #             'split': i
            }
            results.update(eval_dict)
        else:
            results['aucs'] +=  aucs / n_splits
            results['w_auc'] += w_aucs / n_splits
            results['report'] += rep_df / n_splits
            for k, v in eval_dict.items():
                results[k] += v
        i += 1
    
    return results

def gridsearch_cv_xgb(X, y, param_grid, n_splits=5, n_round=200, random_state=444, verbose_eval=True, n_class=10)->list:
    """
    网格搜索以及交叉验证
    Reture
    ----------
    list对象，每个元素是一个字典，字典结构参考 `cv_xgb` 函数
    """
    results = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for i, p in enumerate(param_grid):  # 针对一组超参数进行cv
        t0 = time()
        ret = cv_xgb(p, cv, X, y, n_round=n_round, n_splits=n_splits, stratified=True, 
                     shuffle=True, random_state=444, verbose_eval=verbose_eval, n_class=n_class)
        
        ret['param'] = p
        results.append(ret)

        print(f"{i+1} / {len(param_grid)}: {n_round}-rounds Training finished param={p} ...\t\t({time()-t0:.3f}s)")
    
    return results

def myproduct(*iterables):
    n = len(iterables)
    if n == 0:
        raise ValueError("Input's length is zero!") 
    
    ret = []
    ret.extend([[e] for e in iterables[0].copy()])
    if n == 1:
        return ret

    # 将需要调参的参数进行组合，即笛卡尔乘积。类似于sklearn中的 ParameterGrid
    for k in range(1, n):
        v = iterables[k].copy()
        l = len(ret)
        ret = [ret[i%l].copy() for i in range(len(v) * len(ret))]
        for i, e in enumerate(ret):
            e.append(v[i // l])
    return ret

def compose_param_grid(grid, base):
    items = list(grid.items())
    iterables = [item[1] for item in items]
    keys = [item[0] for item in items]

    ret = myproduct(*iterables)
    com_ps = [dict(zip(keys, e)) for e in ret]


    all_params = [base.copy() for _ in range(len(com_ps))] 
    for i in range(len(com_ps)):
        all_params[i].update(com_ps[i])
        
    return all_params


##################################################转换类函数##################################################
def dict_2_str(d, sort_by_key=True):
    items = list(d.items())
    if sort_by_key:
        items.sort(key=lambda x: x[0])
        
    s = list(map(lambda x: f"{x[0]}={x[1]}", items))
    return ", ".join(s)

def report_2_df(report):
    report = deepcopy(report)
    acc = report['accuracy']
    report['accuracy'] = {'precision': np.nan, 'recall': np.nan, 'f1-score': acc, 'support': report['macro avg']['support']}
    df = pd.DataFrame(report).T
    df['support'] = df['support'].astype(np.int64)
    return df

def eval_str_2_dict(eval_str):
    """将xgboost.Booster的eval方法返回的eval str转换为字典类型"""
    if not eval_str:
        return dict()
    
    evals = eval_str.split('\t')[1:]
    evals = list(map(lambda x: x.split(':'), evals))
    evals = list(map(lambda x: (f"'{x[0]}':{x[1]}"), evals))

    evals = ", ".join(evals)
    evals = eval("{"+evals+"}")
    
    return evals

def metric_2_str(result):
    """将一组参数的评价转换为字符串"""
    result = deepcopy(result)
    param = f"# {result.pop('param', '')}"
    ret = [param]
    
    for k, v in result.items():
        if isinstance(v, pd.DataFrame):
            v = v.to_markdown(tablefmt="grid")
        is_break = '\n' if '\n' in str(v) or len(str(v)) > 100 else ""
        ret.append(f"- {k} :{is_break}{v}")
    
    s = '\n'.join(ret)
    s = f'{s}\n'
    
    return s


# 节约内存的一个标配函数
def reduce_mem(df):
    starttime = time()
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
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df