# 2021-08-24 03:52:28.835092

## Comment: 
特征：基础特征+用户和视频的统计量特征。
数据集划分：watch_label的测试集为.2，is_share的测试集为.2。
watch_label训练250rounds，is_share训练300rounds。
数据均衡：对watch_label中较少的类别（5,6,7,8）使用SMOTE进行了上采样 --- {5: 150000, 6: 140000, 7: 130000, 8: 150000}，对is_share中的1类别进行了上采样到16000。数据集划分时进行了shuffle和stratified。
构造watch_label的xgb.DMatrix时为每个样本设置了权重，指为对于类别所占比例。
此次生成的提交是：../submission-1629748141.csv。官方测评得分：1.640504😐

## model name: ['wl_model_v20.2', 'sh_model_v20.2']
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
- Aucs (watch_label) : [0.66950232 0.59816541 0.50868854 0.50059048 0.50031236 0.50094318
 0.52500593 0.50692615 0.51302808 0.63943336]
- Weighted Aucs (watch_label) : 2.418092285106105
- Aucs (is_share) : [0.67026482 0.67026482]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.36      0.51      0.42     77704
           1       0.25      0.47      0.32     77704
           2       0.20      0.04      0.07     62822
           3       0.20      0.00      0.00     43838
           4       0.25      0.00      0.00     34481
           5       0.30      0.00      0.00     30000
           6       0.35      0.06      0.10     28000
           7       0.30      0.02      0.03     26000
           8       0.33      0.03      0.05     30000
           9       0.25      0.65      0.36     77704

    accuracy                           0.27    488253
   macro avg       0.28      0.18      0.14    488253
weighted avg       0.27      0.27      0.20    488253

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.69      0.63      0.66      3209
           1       0.66      0.71      0.68      3191

    accuracy                           0.67      6400
   macro avg       0.67      0.67      0.67      6400
weighted avg       0.67      0.67      0.67      6400

