# 2021-08-22 11:42:11.894058

## Comment: 
特征：基础特征+用户和视频的统计量特征。
数据集划分：watch_label的测试集为.2，is_share的测试集为.2。
watch_label训练300rounds，is_share训练600rounds。
此次生成的提交是：../submission-1629603492.csv。官方测评得分：1.890610😐
1.890610
## model name: ['wl_model_v18', 'sh_model_v18']
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
- Traing_Data.shape (watch_label)  : (2156311, 127)
- Testing_Data.shape (watch_label) : (239591, 127)
- Traing_Data.shape (is_share)  : (22910, 127)
- Testing_Data.shape (is_share) : (5728, 127)
- Resampled class distribution (watch_label): 
Counter({1: 388521, 9: 388521, 0: 388521, 2: 314107, 3: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({0: 14319, 1: 14319})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.66923905 0.59721769 0.51477513 0.50586486 0.50401369 0.50401172
 0.50597849 0.50252654 0.51054198 0.63821021]
- Weighted Aucs (watch_label) : 2.4062260256445
- Aucs (is_share) : [0.65638599 0.65638599]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.37      0.51      0.43     38973
           1       0.26      0.43      0.32     38841
           2       0.18      0.10      0.13     31297
           3       0.18      0.02      0.04     21689
           4       0.18      0.01      0.02     17005
           5       0.18      0.01      0.02     14485
           6       0.21      0.02      0.03     12694
           7       0.18      0.01      0.01     11846
           8       0.25      0.03      0.05     13820
           9       0.26      0.63      0.36     38941

    accuracy                           0.27    239591
   macro avg       0.22      0.18      0.14    239591
weighted avg       0.24      0.27      0.21    239591

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.66      0.62      0.64      2845
           1       0.65      0.69      0.67      2883

    accuracy                           0.66      5728
   macro avg       0.66      0.66      0.66      5728
weighted avg       0.66      0.66      0.66      5728

