# 2021-08-31 02:14:22.387490

## Comment: 
特征：v1版基础特征+用户和视频的统计量特征，添加了is_happy_day特征，表示这天是否为周末或节假日。
数据集划分：watch_label的测试集为.18，is_share的测试集为.18。
watch_label训练250rounds，早停=30，is_share训练250rounds，早停=60。
。数据集划分时进行了shuffle和stratified。
此次生成的提交是：../submission-1630346741.csv。官方测评得分：xxx😐

## model name: ['', '']
- model save path : /home/gzy/jupyter-lab/multi-objects-video-recommendation/Models

## Data setup
- dataset.shape : (7353024, 128)
- dataset.columns : Index(['v_avg_watch_label_1', 'v_sum_watch_times_1', 'v_sum_watch_overs_1',
       'v_sum_comment_times_1', 'v_sum_collect_times_1', 'v_sum_share_times_1',
       'v_sum_quit_times_1', 'v_sum_skip_times_1', 'v_sum_watch_days_1',
       'v_avg_watch_label_3',
       ...
       'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'da_0', 'da_1',
       'da_2', 'da_3', 'da_4'],
      dtype='object', length=128)
- is resample : True
- Traing_Data.shape (watch_label)  : (22911, 128)
- Testing_Data.shape (watch_label) : (5727, 128)

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.01, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.005, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.65664533 0.58962866 0.50507785 0.50249548 0.50244899 0.50275637
 0.5046687  0.50197719 0.50757588 0.62606703]
- Weighted Aucs (watch_label) : 2.3867911395958608
- Aucs (is_share) : [0.65664533 0.58962866 0.50507785 0.50249548 0.50244899 0.50275637
 0.5046687  0.50197719 0.50757588 0.62606703]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support
0              0.373191  0.447464  0.406966   77704.2
1              0.278011  0.642835  0.388154  111484.2
2              0.261718  0.016755  0.031493   62821.4
3              0.219837  0.007473  0.014454   43837.6
4              0.222941  0.006543  0.012710   34480.8
5              0.210198  0.007084  0.013703   28600.2
6              0.268990  0.010856  0.020862   25018.4
7              0.213552  0.004807  0.009402   23549.8
8              0.329894  0.017147  0.032597   27759.6
9              0.261190  0.509344  0.345307   77704.2
accuracy            NaN       NaN  0.289853  512960.4
macro avg      0.263952  0.167031  0.127565  512960.4
weighted avg   0.274840  0.289853  0.208239  512960.4
- Classification Report (is_share) : 

              precision    recall  f1-score  support
0              0.738168  0.368111  0.491119   2863.8
1              0.579109  0.869334  0.695125   2863.8
accuracy            NaN       NaN  0.618723   5727.6
macro avg      0.658639  0.618723  0.593122   5727.6
weighted avg   0.658638  0.618723  0.593122   5727.6
