# 2021-08-24 03:57:06.491777

## Comment: 
特征：基础特征+用户和视频的统计量特征。
数据集划分：watch_label的测试集为.2，is_share的测试集为.2。
watch_label训练250rounds，is_share训练300rounds。
数据均衡：对watch_label中较少的类别（5,6,7,8）使用SMOTE进行了上采样 --- {5: 150000, 6: 140000, 7: 130000, 8: 150000}，对is_share中的1类别进行了上采样到16000。数据集划分时进行了shuffle和stratified。
此次生成的提交是：../submission-1629748518.csv。官方测评得分：1.894175😐

## model name: ['wl_model_v20.3', 'sh_model_v20.3']
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
- Traing_Data.shape (watch_label)  : (1953009, 127)
- Testing_Data.shape (watch_label) : (488253, 127)
- Traing_Data.shape (is_share)  : (25600, 127)
- Testing_Data.shape (is_share) : (6400, 127)
- Resampled class distribution (watch_label): 
Counter({1: 388521, 9: 388521, 0: 388521, 2: 314107, 3: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({0: 16000, 1: 14319})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.66949492 0.59624247 0.51615496 0.50443174 0.50371586 0.50437625
 0.53264757 0.5150593  0.5191135  0.64129271]
- Weighted Aucs (watch_label) : 2.440443516535658
- Aucs (is_share) : [0.67026482 0.67026482]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.36      0.51      0.42     77704
           1       0.26      0.42      0.32     77704
           2       0.18      0.10      0.13     62822
           3       0.16      0.02      0.03     43838
           4       0.19      0.01      0.02     34481
           5       0.19      0.01      0.02     30000
           6       0.34      0.07      0.12     28000
           7       0.28      0.04      0.06     26000
           8       0.31      0.04      0.08     30000
           9       0.25      0.64      0.36     77704

    accuracy                           0.27    488253
   macro avg       0.25      0.19      0.16    488253
weighted avg       0.26      0.27      0.21    488253

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.69      0.63      0.66      3209
           1       0.66      0.71      0.68      3191

    accuracy                           0.67      6400
   macro avg       0.67      0.67      0.67      6400
weighted avg       0.67      0.67      0.67      6400

