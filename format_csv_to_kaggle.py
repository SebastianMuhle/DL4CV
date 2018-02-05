import utility
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import h5py
import os

learning_data_root = 'data/learning/'
photo_root = learning_data_root + 'photos/'
train_photos_root = photo_root + 'train/'
validation_photos_root = photo_root + 'validation/'

test_data_root = 'data/test/'
test_photo_root = learning_data_root + 'photos/'

# Image Parameters
# Xception 299, 299 - VGG16 224, 224
img_width, img_height = 224, 224
img_shape = (img_width,img_height,3)

#DL Parameters
batch_size = 16

classes = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive',
               'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']
nb_classes = 9

# csv to list in utility.py
results = pd.read_csv(test_data_root+'results.csv')
print(results[0:5])
sample_submission = pd.read_csv(test_data_root+'sample_submission.csv')
print(sample_submission[0:5])
test = pd.merge(results,sample_submission, on="business_id", how="outer")
del test['labels_y']
test.columns = ['business_id','labels']
print(test[0:5])
print(test.shape)
test.to_csv(test_data_root+"kaggle_results.csv",index=False)

# x_biz = list(set(test["business_id"].tolist()))
# del test['test_photo_to_biz.csv']
# test = test.drop_duplicates(keep='first')
# test.to_csv("asdafasdaf.csv",index=False)
# print(test[:5])

# business_to_label_dict = {}
# for business in x_biz:
# 	business_df = df[df['business_id']==business]
# 	if (business_df.empty==False):
# 		predictions = np.asarray(list(business_df['prediction'].values))
# 		prediction = np.around(np.sum(predictions,axis=0)/predictions.shape[0]).tolist()
# 		prediction_str = ''
# 		for i in range(len(prediction)):
# 			if prediction[i] == 1:
# 				prediction_str += str(i)+" "
# 		if prediction_str != '':
# 			prediction_str = prediction_str[:-1]
# 		business_to_label_dict[business] = prediction_str
# 	else:
# 		business_to_label_dict[business] = None
# df = pd.DataFrame(list(business_to_label_dict.items()),columns=['business_id','labels'])


