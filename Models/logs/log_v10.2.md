# 2021-07-06 18:56:00.845057

## Comment: 
将submission-1625555134.csv中的is_share替换为submission-1625404864.csv中的is_share，保留watch_label。
watch_label的预测：基础特征，is_share的预测：基础特征。
此次生成的提交是：submission-1625568173.csv。官方测评得分：1.806136😐

## model name: ['wl_model_v10', '参见log.9.2.md']
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
{'objective': 'multi:softmax', 'eta': 0.1, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'max_depth': 11, 'min_child_weight': 7, 'gamma': 0.4}
- model params (is_share) : 
参见log.9.2.md

## Model's Performance
- Aucs (watch_label) : [0.57555597 0.58567653 0.50347593 0.50017473 0.50017414 0.5000983
 0.50292636 0.50040085 0.50878295 0.60767447]
- Weighted Aucs (watch_label) : 2.365403849959146
- Aucs (is_share) : 参见log.9.2.md
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.33      0.19      0.24     43749
           1       0.29      0.68      0.41    111616
           2       0.23      0.01      0.03     62829
           3       0.13      0.00      0.00     43872
           4       0.15      0.00      0.00     34228
           5       0.18      0.00      0.00     28926
           6       0.33      0.01      0.01     25061
           7       0.16      0.00      0.00     23400
           8       0.31      0.02      0.04     27750
           9       0.24      0.56      0.34     77663

    accuracy                           0.27    479094
   macro avg       0.23      0.15      0.11    479094
weighted avg       0.24      0.27      0.18    479094

- Classification Report (is_share) : 

参见log.9.2.md
