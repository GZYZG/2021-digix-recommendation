# 2021-07-09 15:12:15.525350

## Comment: 
对watch_label和is_share进行网格搜索以及交叉验证，交叉验证的结果均保存为.pkl文件。
watch_label的预测：基础特征，is_share的预测：基础特征。
此次生成的提交是：submission-1625814091.csv。官方测评得分：xxx😐

## model name: ['wl_model_v11', 'sh_model_v11']
- model save path : /home/gzy/jupyter-lab/multi-objects-video-recommendation/Models

## Data setup
- dataset.shape : (7353024, 70)
- dataset.columns : Index(['age_0', 'age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7',
       'gender_0', 'gender_1', 'gender_2', 'gender_3', 'city_level_0',
       'city_level_1', 'city_level_2', 'city_level_3', 'city_level_4',
       'city_level_5', 'city_level_6', 'city_level_7', 'device_name_0',
       'device_name_1', 'device_name_2', 'device_name_3', 'device_name_4',
       'device_name_5', 'device_name_6', 'device_name_7', 'device_name_8',
       'device_name_9', 'video_score', 'video_duration', 'video_release_year',
       'video_release_month', 'video_release_day', 'desc_0', 'desc_1',
       'desc_2', 'desc_3', 'desc_4', 'desc_5', 'desc_6', 'desc_7', 'desc_8',
       'desc_9', 'tags_0', 'tags_1', 'tags_2', 'tags_3', 'tags_4', 'tags_5',
       'tags_6', 'tags_7', 'tags_8', 'tags_9', 'class_0', 'class_1', 'class_2',
       'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8',
       'class_9', 'da_0', 'da_1', 'da_2', 'da_3', 'da_4'],
      dtype='object')
- is resample : True
- Traing_Data.shape (watch_label)  : (1916375, 70)
- Testing_Data.shape (watch_label) : (479094, 70)
- Traing_Data.shape (is_share)  : (23550, 70)
- Testing_Data.shape (is_share) : (5888, 70)
- Resampled class distribution (watch_label): 
Counter({1: 557421, 9: 388521, 2: 314107, 0: 219188, 3: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({0: 15119, 1: 14319})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['mlogloss', 'auc', 'merror'], 'max_depth': 9, 'min_child_weight': 9, 'gamma': 0.2, 'subsample': 0.9, 'colsample_bytree': 0.6, 'reg_alpha': 0}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'nthread': 8, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'eval_metric': ['logloss', 'auc', 'error'], 'max_depth': 5, 'min_child_weight': 1, 'gamma': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 0}

## Model's Performance
- Aucs (watch_label) : [0.56775223 0.58589565 0.50352267 0.50013574 0.50020896 0.50015228
 0.5028023  0.50020022 0.50903426 0.60518875]
- Weighted Aucs (watch_label) : 2.36321336447277
- Aucs (is_share) : [0.6076795 0.6076795]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.35      0.17      0.23     44017
           1       0.29      0.70      0.41    111575
           2       0.26      0.01      0.02     62786
           3       0.13      0.00      0.00     43844
           4       0.22      0.00      0.00     34491
           5       0.23      0.00      0.00     28448
           6       0.30      0.01      0.01     25054
           7       0.14      0.00      0.00     23498
           8       0.30      0.02      0.04     27589
           9       0.24      0.56      0.33     77792

    accuracy                           0.27    479094
   macro avg       0.24      0.15      0.10    479094
weighted avg       0.25      0.27      0.18    479094

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.64      0.57      0.60      3073
           1       0.58      0.64      0.61      2815

    accuracy                           0.61      5888
   macro avg       0.61      0.61      0.61      5888
weighted avg       0.61      0.61      0.61      5888

