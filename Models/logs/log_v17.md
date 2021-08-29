# 2021-08-21 21:13:18.760398

## Comment: 
特征：基础特征+用户和视频的统计量特征。
数据集划分：watch_label的测试集为1.0，is_share的测试集为1.0。
此次生成的提交是：submission-1629551361.csv。官方测评得分：1.899124😐，is_share预测结果好于v14的is_share。

## model name: ['wl_model_v17', 'sh_model_v17']
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
- Traing_Data.shape (watch_label)  : (2395901, 127)
- Testing_Data.shape (watch_label) : (1, 127)
- Traing_Data.shape (is_share)  : (28637, 127)
- Testing_Data.shape (is_share) : (1, 127)
- Resampled class distribution (watch_label): 
Counter({1: 388521, 0: 388521, 9: 388521, 2: 314107, 3: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({0: 14319, 1: 14319})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.  0.5 0.  0.  0.  0.  0.  0.  0.  0. ]
- Weighted Aucs (watch_label) : 0.05
- Aucs (is_share) : [0.  0.5]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           1       1.00      1.00      1.00         1

    accuracy                           1.00         1
   macro avg       1.00      1.00      1.00         1
weighted avg       1.00      1.00      1.00         1

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           1       1.00      1.00      1.00         1

    accuracy                           1.00         1
   macro avg       1.00      1.00      1.00         1
weighted avg       1.00      1.00      1.00         1

