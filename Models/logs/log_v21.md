# 2021-08-24 23:21:26.600436

## Comment: 
特征：基础特征+用户和视频的统计量特征，添加了is_happy_day特征，表示这天是否为周末或节假日。
数据集划分：watch_label的测试集为.18，is_share的测试集为.18。
watch_label训练250rounds，is_share训练250rounds。
。数据集划分时进行了shuffle和stratified。
此次生成的提交是：../submission-1629818308.csv。官方测评得分：xxx😐

## model name: ['wl_model_v21', 'sh_model_v21']
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
- Traing_Data.shape (watch_label)  : (1964639, 128)
- Testing_Data.shape (watch_label) : (431263, 128)
- Traing_Data.shape (is_share)  : (23483, 128)
- Testing_Data.shape (is_share) : (5155, 128)
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
- Aucs (watch_label) : [0.6658016  0.59638668 0.51565381 0.5046615  0.50378221 0.50417406
 0.50618541 0.50343793 0.51154374 0.63719436]
- Weighted Aucs (watch_label) : 2.406595518037825
- Aucs (is_share) : [0.64919658 0.64919658]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.36      0.51      0.42     69934
           1       0.26      0.43      0.32     69934
           2       0.18      0.10      0.13     56539
           3       0.16      0.02      0.03     39454
           4       0.19      0.01      0.02     31033
           5       0.19      0.01      0.02     25740
           6       0.23      0.02      0.03     22516
           7       0.20      0.01      0.02     21195
           8       0.28      0.03      0.05     24984
           9       0.26      0.63      0.36     69934

    accuracy                           0.27    431263
   macro avg       0.23      0.18      0.14    431263
weighted avg       0.24      0.27      0.21    431263

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.68      0.57      0.62      2575
           1       0.63      0.73      0.67      2580

    accuracy                           0.65      5155
   macro avg       0.65      0.65      0.65      5155
weighted avg       0.65      0.65      0.65      5155

