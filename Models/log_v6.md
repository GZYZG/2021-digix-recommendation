# 2021-07-01 21:19:17.554570

## model name: ['wl_model_v6', 'sh_model_v6']
- model save path : /home/gzy/jupyter-lab/multi-objects-video-recommendation/Models

## Data setup
- dataset.shape : (7353024, 72)
- dataset.columns : Index(['user_id', 'video_id', 'age_0', 'age_1', 'age_2', 'age_3', 'age_4',
       'age_5', 'age_6', 'age_7', 'gender_0', 'gender_1', 'gender_2',
       'gender_3', 'city_level_0', 'city_level_1', 'city_level_2',
       'city_level_3', 'city_level_4', 'city_level_5', 'city_level_6',
       'city_level_7', 'device_name_0', 'device_name_1', 'device_name_2',
       'device_name_3', 'device_name_4', 'device_name_5', 'device_name_6',
       'device_name_7', 'device_name_8', 'device_name_9', 'video_score',
       'video_duration', 'video_release_year', 'video_release_month',
       'video_release_day', 'desc_0', 'desc_1', 'desc_2', 'desc_3', 'desc_4',
       'desc_5', 'desc_6', 'desc_7', 'desc_8', 'desc_9', 'tags_0', 'tags_1',
       'tags_2', 'tags_3', 'tags_4', 'tags_5', 'tags_6', 'tags_7', 'tags_8',
       'tags_9', 'class_0', 'class_1', 'class_2', 'class_3', 'class_4',
       'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'da_0', 'da_1',
       'da_2', 'da_3', 'da_4'],
      dtype='object')
- is resample : True
- Traing_Data.shape (watch_label)  : (1916375, 72)
- Testing_Data.shape (watch_label) : (479094, 72)
- Traing_Data.shape (is_share)  : (23550, 72)
- Testing_Data.shape (is_share) : (5888, 72)
- Resampled class distribution (watch_label): 
Counter({1: 557421, 9: 388521, 2: 314107, 0: 219188, 3: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({0: 15119, 1: 14319})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'max_depth': 11, 'min_child_weight': 7, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist'}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'max_depth': 6, 'nthread': 4, 'gpu_id': 0, 'tree_method': 'gpu_hist'}

## Model's Performance
- Aucs (watch_label) : [0.57749827 0.58572151 0.50378435 0.50053127 0.50040545 0.50016001
 0.50280881 0.50046443 0.50977498 0.60535577]
- Weighted Aucs (watch_label) : 2.364381155534087
- Aucs (is_share) : [0.63001847 0.63001847]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.33      0.20      0.25     43915
           1       0.29      0.67      0.40    111680
           2       0.19      0.02      0.04     62504
           3       0.12      0.00      0.01     43621
           4       0.12      0.00      0.00     34605
           5       0.11      0.00      0.00     28676
           6       0.27      0.01      0.01     25168
           7       0.12      0.00      0.00     23377
           8       0.29      0.02      0.04     27486
           9       0.24      0.56      0.33     78062

    accuracy                           0.27    479094
   macro avg       0.21      0.15      0.11    479094
weighted avg       0.22      0.27      0.18    479094

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.65      0.61      0.63      3058
           1       0.61      0.65      0.63      2830

    accuracy                           0.63      5888
   macro avg       0.63      0.63      0.63      5888
weighted avg       0.63      0.63      0.63      5888

