# 2021-08-30 03:18:42.264974

## Comment: 
特征：v1版基础特征+用户和视频的统计量特征，添加了is_happy_day特征，表示这天是否为周末或节假日。
数据集划分：watch_label的测试集为.18，is_share的测试集为.18。
watch_label训练250rounds，早停=30，is_share训练250rounds，早停=60。
。数据集划分时进行了shuffle和stratified。
此次生成的提交是：../submission-1630265746.csv。官方测评得分：1.831257😐

## model name: ['wl_model_v23', 'sh_model_v23']
- model save path : /home/gzy/jupyter-lab/multi-objects-video-recommendation/Models

## Data setup

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.66663234 0.59726837 0.5154675  0.50434376 0.50359383 0.50423387
 0.50620546 0.50287012 0.5098358  0.63711319]
- Weighted Aucs (watch_label) : 2.4046808020465744
- Aucs (is_share) : [0.65876682 0.65876682]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.36      0.51      0.42     69934
           1       0.26      0.43      0.32     69934
           2       0.18      0.10      0.13     56539
           3       0.16      0.02      0.03     39454
           4       0.17      0.01      0.02     31033
           5       0.19      0.01      0.02     25740
           6       0.24      0.02      0.03     22516
           7       0.18      0.01      0.01     21195
           8       0.26      0.02      0.04     24984
           9       0.25      0.63      0.36     69934

    accuracy                           0.27    431263
   macro avg       0.23      0.18      0.14    431263
weighted avg       0.24      0.27      0.21    431263

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.68      0.59      0.63      2673
           1       0.64      0.73      0.68      2709

    accuracy                           0.66      5382
   macro avg       0.66      0.66      0.66      5382
weighted avg       0.66      0.66      0.66      5382

