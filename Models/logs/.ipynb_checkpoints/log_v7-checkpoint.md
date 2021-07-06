# 2021-07-04 16:18:55.387613

## Comment: 
模型配置与log_v6.md中的配置一致，但是训练数据中剔除了未在视频特征中出现的视频数据。
此次生成的提交是：submission-1625380995.csv。官方测评得分：1.846189😁

## model name: ['wl_model_v7', 'sh_model_v7']
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
- Traing_Data.shape (watch_label)  : (1903695, 70)
- Testing_Data.shape (watch_label) : (475924, 70)
- Traing_Data.shape (is_share)  : (23465, 70)
- Testing_Data.shape (is_share) : (5867, 70)
- Resampled class distribution (watch_label): 
Counter({1: 554320, 9: 385082, 2: 312266, 3: 217820, 0: 217820, 4: 171292, 5: 142012, 8: 137834, 6: 124245, 7: 116928})
- Resampled class distribution (is_share): 
Counter({0: 15066, 1: 14266})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'max_depth': 11, 'min_child_weight': 7, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist'}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'max_depth': 6, 'nthread': 4, 'gpu_id': 0, 'tree_method': 'gpu_hist'}

## Model's Performance
- Aucs (watch_label) : [0.57475393 0.5871563  0.50416606 0.50046542 0.50023603 0.50015738
 0.50279737 0.5004294  0.50918619 0.60538307]
- Weighted Aucs (watch_label) : 2.3640342890978734
- Aucs (is_share) : [0.61873752 0.61873752]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.32      0.19      0.24     43610
           1       0.29      0.68      0.41    110771
           2       0.21      0.02      0.03     62640
           3       0.12      0.00      0.01     43273
           4       0.11      0.00      0.00     33903
           5       0.11      0.00      0.00     28635
           6       0.28      0.01      0.01     24985
           7       0.14      0.00      0.00     23501
           8       0.29      0.02      0.04     27429
           9       0.24      0.56      0.33     77177

    accuracy                           0.27    475924
   macro avg       0.21      0.15      0.11    475924
weighted avg       0.23      0.27      0.18    475924

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.64      0.59      0.62      3019
           1       0.60      0.64      0.62      2848

    accuracy                           0.62      5867
   macro avg       0.62      0.62      0.62      5867
weighted avg       0.62      0.62      0.62      5867

