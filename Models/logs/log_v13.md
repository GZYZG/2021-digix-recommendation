# 2021-08-20 19:47:55.578663

## Comment: 
增加了用户和视频的统计量特征。
此次生成的提交是：submission-1629459856.csv。官方测评得分：1.884339😐

## model name: ['wl_model_v13', 'sh_model_v13']
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
- Traing_Data.shape (watch_label)  : (1916375, 127)
- Testing_Data.shape (watch_label) : (479094, 127)
- Traing_Data.shape (is_share)  : (23550, 127)
- Testing_Data.shape (is_share) : (5888, 127)
- Resampled class distribution (watch_label): 
Counter({1: 557421, 9: 388521, 2: 314107, 3: 219188, 0: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({0: 15119, 1: 14319})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.6123405  0.59051761 0.50552355 0.50286763 0.50254482 0.50287541
 0.50546444 0.5026463  0.50882701 0.63340024]
- Weighted Aucs (watch_label) : 2.3957252885038796
- Aucs (is_share) : [0.64404932 0.64404932]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.37      0.27      0.31     43826
           1       0.29      0.70      0.41    111650
           2       0.22      0.02      0.04     62961
           3       0.19      0.01      0.02     43774
           4       0.19      0.01      0.01     34121
           5       0.19      0.01      0.02     28808
           6       0.28      0.01      0.02     24783
           7       0.22      0.01      0.01     23594
           8       0.30      0.02      0.04     27740
           9       0.27      0.56      0.36     77837

    accuracy                           0.29    479094
   macro avg       0.25      0.16      0.13    479094
weighted avg       0.26      0.29      0.20    479094

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.69      0.57      0.62      3066
           1       0.61      0.71      0.66      2822

    accuracy                           0.64      5888
   macro avg       0.65      0.64      0.64      5888
weighted avg       0.65      0.64      0.64      5888

