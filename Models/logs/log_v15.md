# 2021-08-21 18:20:41.842029

## Comment: 
特征：基础特征+用户和视频的统计量特征。
数据集划分：watch_label的测试集为0.1，is_share的测试集为0.2，但0类样本的数目比1少800。
此次生成的提交是：submission-1629468285.csv。官方测评得分：1.896272😐

## model name: ['wl_model_v15', 'sh_model_v15']
- model save path : /home/gzy/jupyter-lab/multi-objects-vid1.896272eo-recommendation/Models

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
- Traing_Data.shape (is_share)  : (22270, 127)
- Testing_Data.shape (is_share) : (5568, 127)
- Resampled class distribution (watch_label): 
Counter({1: 388521, 9: 388521, 0: 388521, 2: 314107, 3: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({1: 14319, 0: 13519})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.66737327 0.59639346 0.51560779 0.50401663 0.50307317 0.50397848
 0.50659913 0.50302412 0.50841231 0.63653613]
- Weighted Aucs (watch_label) : 2.4028731334626015
- Aucs (is_share) : [0.63834812 0.63834812]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.36      0.51      0.42     38754
           1       0.26      0.43      0.32     38913
           2       0.18      0.10      0.13     31420
           3       0.16      0.02      0.03     21822
           4       0.17      0.01      0.02     17210
           5       0.19      0.01      0.02     14326
           6       0.24      0.02      0.03     12532
           7       0.19      0.01      0.01     11760
           8       0.25      0.02      0.04     13923
           9       0.25      0.63      0.36     38931

    accuracy                           0.27    239591
   macro avg       0.23      0.18      0.14    239591
weighted avg       0.24      0.27      0.21    239591

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.68      0.49      0.57      2706
           1       0.62      0.78      0.69      2862

    accuracy                           0.64      5568
   macro avg       0.65      0.64      0.63      5568
weighted avg       0.65      0.64      0.63      5568

