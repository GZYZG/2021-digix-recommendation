# 2021-08-22 16:51:56.820055

## Comment: 
特征：基础特征+用户和视频的统计量特征，除此之外，is_share的特征还加上了watch_label，训练集使用的真实的watch_label，测试集使用预测的watch_label。
数据集划分：watch_label的测试集为.15，is_share的测试集为.2。
watch_label训练300rounds，is_share训练600rounds。
此次生成的提交是：../submission-1629622213.csv。官方测评得分：1.887021😐

## model name: ['wl_model_v19', 'sh_model_v19']
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
- Traing_Data.shape (watch_label)  : (2036516, 127)
- Testing_Data.shape (watch_label) : (359386, 127)
- Traing_Data.shape (is_share)  : (22910, 128)
- Testing_Data.shape (is_share) : (5728, 128)
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
- Aucs (watch_label) : [0.6676927  0.59549384 0.51434806 0.50466707 0.5041683  0.50455988
 0.50658242 0.50320763 0.51051575 0.63840191]
- Weighted Aucs (watch_label) : 2.4069354911900125
- Aucs (is_share) : [0.66602734 0.66602734]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.36      0.51      0.42     58076
           1       0.26      0.43      0.32     58168
           2       0.18      0.10      0.13     46949
           3       0.17      0.02      0.03     33008
           4       0.19      0.01      0.02     26050
           5       0.20      0.01      0.02     21748
           6       0.23      0.02      0.03     18650
           7       0.19      0.01      0.02     17723
           8       0.24      0.03      0.05     20595
           9       0.26      0.63      0.37     58419

    accuracy                           0.27    359386
   macro avg       0.23      0.18      0.14    359386
weighted avg       0.24      0.27      0.21    359386

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.68      0.63      0.65      2877
           1       0.65      0.70      0.68      2851

    accuracy                           0.67      5728
   macro avg       0.67      0.67      0.67      5728
weighted avg       0.67      0.67      0.67      5728

