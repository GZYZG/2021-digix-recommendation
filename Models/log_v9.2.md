# 2021-07-04 21:23:08.333969

## Comment: 
模型配置与log_v8.md中的配置一致，使用的完整的视频特征（后续不加特殊说明均为完整特征）。
watch_label的预测：基础特征，is_share的预测：基础特征。
此次生成的提交是：submission-1625404864.csv。官方测评得分：1.847921😐

## model name: ['wl_model_v9', 'sh_model_v9']
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
Counter({1: 557421, 9: 388521, 2: 314107, 3: 219188, 0: 219188, 4: 172404, 5: 143001, 8: 138798, 6: 125092, 7: 117749})
- Resampled class distribution (is_share): 
Counter({0: 15119, 1: 14319})

## Model Params
- model params (watch_label) : 
{'objective': 'multi:softmax', 'eta': 0.1, 'max_depth': 11, 'min_child_weight': 7, 'nthread': 8, 'num_class': 10, 'gpu_id': 0, 'tree_method': 'gpu_hist'}
- model params (is_share) : 
{'objective': 'binary:hinge', 'eta': 0.1, 'max_depth': 6, 'nthread': 4, 'gpu_id': 0, 'tree_method': 'gpu_hist'}

## Model's Performance
- Aucs (watch_label) : [0.57430505 0.58606509 0.50357122 0.50041816 0.50033007 0.50013865
 0.50273403 0.50042983 0.50980292 0.60638754]
- Weighted Aucs (watch_label) : 2.3651799768885837
- Aucs (is_share) : [0.6125295 0.6125295]
- Classification Report (watch_label) : 

              precision    recall  f1-score   support

           0       0.87      0.54      0.67     43867
           1       0.41      0.73      0.52    111843
           2       0.41      0.16      0.23     62804
           3       0.46      0.13      0.20     43875
           4       0.45      0.09      0.15     34453
           5       0.55      0.10      0.17     28592
           6       0.46      0.05      0.08     24837
           7       0.56      0.07      0.12     23675
           8       0.39      0.07      0.12     27616
           9       0.31      0.75      0.43     77532

    accuracy                           0.40    479094
   macro avg       0.49      0.27      0.27    479094
weighted avg       0.46      0.40      0.34    479094

- Classification Report (is_share) : 

              precision    recall  f1-score   support

           0       0.63      0.59      0.61      3053
           1       0.59      0.63      0.61      2835

    accuracy                           0.61      5888
   macro avg       0.61      0.61      0.61      5888
weighted avg       0.61      0.61      0.61      5888

