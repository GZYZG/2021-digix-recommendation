# 2021-08-30 01:57:25.930352

## Comment: 
特征：v1版基础特征+用户和视频的统计量特征，添加了is_happy_day特征，表示这天是否为周末或节假日。
数据集划分：watch_label的测试集为.18，is_share的测试集为.18。
watch_label训练250rounds，早停=30，is_share训练250rounds，早停=60。
。数据集划分时进行了shuffle和stratified。
此次生成的提交是：../submission-1630259438.csv。官方测评得分：xxx😐

## model name: ['wl_model_v22', 'sh_model_v22']
- model save path : /home/gzy/jupyter-lab/multi-objects-video-recommendation/Models

## Data setup

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : 
- Weighted Aucs (watch_label) : 
- Aucs (is_share) : 
- Classification Report (watch_label) : 


- Classification Report (is_share) : 


