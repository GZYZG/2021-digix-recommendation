# 2021-08-23 22:05:55.217819

## Comment: 
特征：基础特征+用户和视频的统计量特征。
数据集划分：watch_label的测试集为.15，is_share的测试集为.2。
watch_label训练300rounds，is_share训练300rounds。
数据均衡：对watch_label中较少的类别（5,6,7,8）使用SMOTE进行了上采样，对is_share中的1类别进行了上采样。
此次生成的提交是：../submission-1629726004.csv。官方测评得分：1.883228😐

## model name: ['wl_model_v20', 'sh_model_v20']
- model save path : /home/gzy/jupyter-lab/multi-objects-video-recommendation/Models

## Data setup
- dataset.shape : (7353024, 127)
- dataset.columns : Index(['v_avg_watch_label_1', 'v_sum_watch_times_1', 'v_sum_watch_overs_1',
       'v_sum_comment_times_1', 'v_sum_collect_times_1', 'v_sum_share_times_1',
       'v_sum_quit_times_1', 'v_sum_skip_times_1', 'v_sum_watch_days_1',
       'v_avg_watch_label_3',
       ...
       'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'da_0', 'da_1',
       'da_2', 'da_3', 'da_4'],
      dtype='object', length=127)
- is resample : True
- Traing_Data.shape (watch_label)  : (2270572, 127)
- Testing_Data.shape (watch_label) : (400690, 127)
- Traing_Data.shape (is_share)  : (32000, 127)
- Testing_Data.shape (is_share) : (8000, 127)
- Resampled class distribution (watch_label): 
Counter({0: 388521, 1: 388521, 9: 388521, 2: 314107, 3: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({0: 20000, 1: 14319})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.67920077 0.61067432 0.52032877 0.50515452 0.50472049 0.52284957
 0.57331587 0.60890579 0.54043595 0.65710711]
- Weighted Aucs (watch_label) : 2.573961246746091
- Aucs (is_share) : [0.71507731 0.71507731]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.36      0.51      0.42     58318
           1       0.26      0.43      0.32     57979
           2       0.18      0.10      0.13     47166
           3       0.16      0.02      0.03     33082
           4       0.20      0.01      0.02     25947
           5       0.34      0.05      0.09     29921
           6       0.37      0.17      0.23     29999
           7       0.38      0.25      0.30     30116
           8       0.37      0.09      0.15     29903
           9       0.25      0.63      0.36     58259

    accuracy                           0.29    400690
   macro avg       0.29      0.23      0.21    400690
weighted avg       0.28      0.29      0.24    400690

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.70      0.75      0.72      3991
           1       0.73      0.68      0.71      4009

    accuracy                           0.71      8000
   macro avg       0.72      0.72      0.71      8000
weighted avg       0.72      0.71      0.71      8000

