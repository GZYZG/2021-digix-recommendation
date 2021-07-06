from copy import deepcopy
import pandas as pd
import numpy as np
from datatable import join
import datatable as dt


def report_2_df(report):
    report = deepcopy(report)
    acc = report['accuracy']
    report['accuracy'] = {'precision': np.nan, 'recall': np.nan, 'f1-score': acc, 'support': report['macro avg']['support']}
    df = pd.DataFrame(report).T
    df['support'] = df['support'].astype(np.int64)
    return df


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


def cv_xgb(param, cv, X, y, n_round=200, n_class=10, n_splits=5, stratified=True, shuffle=True, random_state=444, verbose_eval=True):
    if isinstance(cv, int):
        if stratified:
            cv = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
        else:
            cv = KFold(n_splits, shuffle=shuffle, random_state=random_state)
    
    results = []
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    
    num_round = n_round
    i = 0
    for train_idxs, test_idxs in cv.split(X, y):
        X_train = X[train_idxs]
        X_test  = X[test_idxs]
        y_train = y[train_idxs]
        y_test  = y[test_idxs]
        
        xg_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        xg_test = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        
        
        t0 = time()
        model = xgb.train(param, xg_train, num_round, watchlist, verbose_eval=verbose_eval)
#         print(f"{num_round}-rounds Training finished ...\t\t({time()-t0:.3f}s)")

        # get prediction
        pred = model.predict(xg_test)
        # pred = pred.astype(np.uint8)
        error_rate = np.sum(pred != y_test) / y_test.shape[0]
        print('Test error using softmax = {}'.format(error_rate))

        
        aucs = auc(y_test.astype(np.uint8), pred.astype(np.uint8), np.arange(n_class))
        # aucs[aucs == 0.5] = 0
        if n_class == 2:
            w_aucs = None
        else:
            weights = np.arange(0, 1, 0.1)
            w_aucs = (aucs * weights).sum()


        rep = metrics.classification_report(list(y_test), list(pred), output_dict=True)
        results.append({
            'test_error': error_rate, 
            'aucs': aucs,
            'w_auc': w_aucs,
            'report': rep,
#             'model': model,
            'split': i
        })
        i += 1
        
    return results


def gridsearch_xgb(grid_params, xg_train, xg_test, num_round=200, n_class=10, verbose_eval=True):
    """
    xg_train, xg_test: DMatrix
    """
    results = []
    num_round = num_round
    n_param = len(grid_params)
    for i, p in enumerate(grid_params):
        t0 = time()
        model = xgb.train(p, xg_train, num_round, watchlist, verbose_eval=verbose_eval)
        print(f"{num_round}-rounds Training finished ...\t\t({time()-t0:.3f}s)")

        # get prediction
        pred = model.predict(xg_test)
        # pred = pred.astype(np.uint8)
        labels = xg_test.get_label()
        error_rate = np.sum(pred != labels) / labels.shape[0]
        print('Test error using softmax = {}'.format(error_rate))

        aucs = auc(labels.astype(np.uint8), pred.astype(np.uint8), np.arange(n_class))
        if n_class == 2:
            w_aucs = None
        else:
            weights = np.arange(0, 1, 0.1)
            w_aucs = (aucs * weights).sum()

        rep = metrics.classification_report(list(labels), list(pred))
        results.append({
            'test_error': error_rate, 
            'aucs': aucs,
            'w_auc': w_aucs,
            'report': rep,
            'model': model
        })

        print(f"{i} / {n_param} : {num_round}-rounds Training finished ...\t\t({time()-t0:.3f}s)")
    
    return results


def gridsearch_cv_xgb(X, y, param_grid, n_splits=5, n_round=200, random_state=444, verbose_eval=True, n_class=10):
    results = []
    num_rounds = n_round
    for i, p in enumerate(param_grid):  # 针对一组超参数进行cv
        t0 = time()
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        ret = cv_xgb(p, cv, X, y, n_round=num_rounds, n_splits=n_splits, stratified=True, 
                     shuffle=True, random_state=444, verbose_eval=verbose_eval, n_class=n_class)
        
        ret.insert(0, p)
        results.append(ret)

        print(f"{i} : {num_round}-rounds Training finished param={p} ...\t\t({time()-t0:.3f}s)")
    
    return results


def myproduct(*iterables):
    n = len(iterables)
    if n == 0:
        return None
    if n == 1:
        return iterables
    
    ret = []
    ret.extend([[e] for e in iterables[0].copy()])

    # 将需要调参的参数进行组合，即笛卡尔乘积。类似于sklearn中的 ParameterGrid
    for k in range(1, n):
        v = iterables[k].copy()
        l = len(ret)
        ret = [ret[i%l].copy() for i in range(len(v) * len(ret))]
        for i, e in enumerate(ret):
            e.append(v[i // l])
    
    return ret