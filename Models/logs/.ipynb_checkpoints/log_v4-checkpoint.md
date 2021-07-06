# 2021-06-30 18:10:20.538894

## model name: ['wl_model_v4', 'sh_model_v4']
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
- Traing_Data.shape (watch_label)  : (2037504, 70)
- Testing_Data.shape (watch_label) : (509377, 70)
- Traing_Data.shape (is_share)  : (22825, 70)
- Testing_Data.shape (is_share) : (5707, 70)
- Resampled class distribution (watch_label): 
Counter({1: 554320, 9: 385082, 0: 385082, 2: 312266, 3: 217820, 4: 171292, 5: 142012, 8: 137834, 6: 124245, 7: 116928})
- Resampled class distribution (is_share): 
Counter({0: 14266, 1: 14266})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'max_depth': 8, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist'}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'max_depth': 7, 'nthread': 4, 'gpu_id': 0, 'tree_method': 'gpu_hist'}

## Model's Performance
- Aucs (watch_label) : [0.62004039 0.58891123 0.50362896 0.50026904 0.50024575 0.50008881
 0.50302352 0.50024341 0.50804977 0.59886517]
- Weighted Aucs (watch_label) : 2.357243306851747
- Aucs (is_share) : [0.62160596 0.62160596]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.32      0.38      0.35     77054
           1       0.28      0.63      0.39    110718
           2       0.28      0.01      0.02     62540
           3       0.15      0.00      0.00     43797
           4       0.21      0.00      0.00     33993
           5       0.14      0.00      0.00     28208
           6       0.30      0.01      0.01     24911
           7       0.12      0.00      0.00     23415
           8       0.28      0.02      0.04     27510
           9       0.23      0.48      0.31     77231

    accuracy                           0.27    509377
   macro avg       0.23      0.15      0.11    509377
weighted avg       0.25      0.27      0.19    509377

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.62      0.59      0.60      2804
           1       0.62      0.66      0.64      2903

    accuracy                           0.62      5707
   macro avg       0.62      0.62      0.62      5707
weighted avg       0.62      0.62      0.62      5707

