# 2021-06-29 23:58:25.913393

## model name: ['wl_model_v3', 'sh_model_v3']
- model save path : /home/gzy/jupyter-lab/multi-objects-video-recommendation/Models

## Data setup
- dataset.shape : (7308018, 70)
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
- Traing_Data.shape (watch_label)  : (1971734, 70)
- Testing_Data.shape (watch_label) : (492934, 70)
- Traing_Data.shape (is_share)  : (22816, 70)
- Testing_Data.shape (is_share) : (5704, 70)
- Resampled class distribution (watch_label): 
Counter({1: 554320, 9: 385082, 2: 312266, 0: 302869, 3: 217820, 4: 171292, 5: 142012, 8: 137834, 6: 124245, 7: 116928})
- Resampled class distribution (is_share): 
Counter({1: 14266, 0: 14254})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'max_depth': 8, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist'}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'max_depth': 6, 'nthread': 4, 'gpu_id': 0, 'tree_method': 'gpu_hist'}

## Model's Performance
- Aucs (watch_label) : [0.59772762 0.58489796 0.50334059 0.5001315  0.50015731 0.50007303
 0.50316653 0.5003208  0.50896805 0.59955772]
- Weighted Aucs (watch_label) : 2.3581976737371133
- Aucs (is_share) : [0.60912243 0.60912243]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.33      0.27      0.30     60667
           1       0.28      0.67      0.39    110396
           2       0.28      0.01      0.02     62526
           3       0.13      0.00      0.00     43744
           4       0.17      0.00      0.00     34450
           5       0.13      0.00      0.00     28316
           6       0.32      0.01      0.01     24744
           7       0.19      0.00      0.00     23339
           8       0.31      0.02      0.04     27467
           9       0.23      0.51      0.32     77285

    accuracy                           0.27    492934
   macro avg       0.24      0.15      0.11    492934
weighted avg       0.25      0.27      0.18    492934

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.62      0.57      0.59      2845
           1       0.60      0.65      0.62      2859

    accuracy                           0.61      5704
   macro avg       0.61      0.61      0.61      5704
weighted avg       0.61      0.61      0.61      5704

