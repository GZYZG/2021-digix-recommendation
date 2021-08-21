# 2021-08-20 22:07:11.463951

## Comment: 
特征：基础特征+用户和视频的统计量特征。
采样：改变了处理类别不均衡时的下采样阈值。
此次生成的提交是：submission-1629468285.csv。官方测评得分：1.900261😐

## model name: ['wl_model_v14', 'sh_model_v14']
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
- Traing_Data.shape (watch_label)  : (1916721, 127)
- Testing_Data.shape (watch_label) : (479181, 127)
- Traing_Data.shape (is_share)  : (22910, 127)
- Testing_Data.shape (is_share) : (5728, 127)
- Resampled class distribution (watch_label): 
Counter({0: 388521, 9: 388521, 1: 388521, 2: 314107, 3: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({0: 14319, 1: 14319})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.66677238 0.59525535 0.51511794 0.5050034  0.50371047 0.50440337
 0.50630927 0.50274627 0.51017652 0.63690281]
- Weighted Aucs (watch_label) : 2.404797713190871
- Aucs (is_share) : [0.64322114 0.64322114]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.36      0.51      0.42     77640
           1       0.26      0.43      0.32     77620
           2       0.18      0.10      0.13     62734
           3       0.18      0.02      0.03     44041
           4       0.19      0.01      0.02     34699
           5       0.20      0.01      0.02     28736
           6       0.24      0.02      0.03     24826
           7       0.19      0.01      0.01     23407
           8       0.28      0.02      0.04     27865
           9       0.25      0.64      0.36     77613

    accuracy                           0.27    479181
   macro avg       0.23      0.18      0.14    479181
weighted avg       0.25      0.27      0.21    479181

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.69      0.53      0.60      2902
           1       0.61      0.75      0.67      2826

    accuracy                           0.64      5728
   macro avg       0.65      0.64      0.64      5728
weighted avg       0.65      0.64      0.64      5728

