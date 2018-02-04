import utility
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import load_model
from InceptionV3model import InceptionV3model
from XceptionModel import XCeptionModel
from VGG16 import VGG16Model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import h5py
from MultilabelGenerator import MultilabelGenerator

learning_data_root = 'data/learning/'
test_data_root = 'data/test/'
models_root = learning_data_root + 'models/'
photo_root = test_data_root + 'photos/'

def prediction_to_df(photo_to_prediction_dict,photo_to_business_dict,x_biz):
	df = pd.DataFrame(list(photo_to_prediction_dict.items()),columns=['photo_id','prediction'])
	df2 = pd.DataFrame(list(photo_to_business_dict.items()),columns=['photo_id','business_id'])
	df = pd.merge(df,df2, on='photo_id')
	x_biz = list(set(x_biz))
	business_to_label_dict = {}
	for business in x_biz:
		business_df = df[df['business_id']==business]
		if (business_df.empty==False):
			predictions = np.asarray(list(business_df['prediction'].values))
			prediction = np.around(np.sum(predictions,axis=0)/predictions.shape[0]).tolist()
			prediction_str = ''
			for i in range(len(prediction)):
				if prediction[i] == 1:
					prediction_str += str(i)+" "
			if prediction_str != '':
				prediction_str = prediction_str[:-1]
			business_to_label_dict[business] = prediction_str
		else:
			business_to_label_dict[business] = None
	df = pd.DataFrame(list(business_to_label_dict.items()),columns=['business_id','labels'])
	return df
	



# Image Parameters
# Xception 299, 299 - VGG16 224, 224
img_width, img_height = 299, 299
img_shape = (img_width,img_height,3)

#DL Parameters
batch_size = 5

classes = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive',
               'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']
nb_classes = 9


# csv to list in utility.py
test_photo_business = pd.read_csv(test_data_root+'test_photo_to_biz.csv')

x_test = np.asarray(test_photo_business['photo_id'].tolist())
x_biz = np.asarray(test_photo_business['business_id'].tolist())

photo_to_business_dict = dict(zip(x_test.tolist(),x_biz.tolist()))

# Test data
test_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=utility.preprocess_input)

utility.apply_mean(test_datagen)

test_multilabel_datagen = MultilabelGenerator(photo_root,
                                    test_datagen,
                                    None,
                                    batch_size=batch_size,
                                    target_size=(img_width,img_height),
                                    train_or_valid='test')

test_generator = test_multilabel_datagen.flow()

# Hyperparameters
num_freezed_layers_array =[132]
learning_rates = [0.0005]

# Hyperparameter search
for num_freezed_layers in num_freezed_layers_array:
	for lr in learning_rates:

		optimizerAdam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,)

		filepath=models_root+"weights_xception.hdf5"

		model = XCeptionModel().create_model(num_freezedLayers=num_freezed_layers, nb_classes=nb_classes,
                                                optimizer=optimizerAdam)

		model.load_weights(filepath)

		predictions = model.predict_generator(test_generator, len(test_multilabel_datagen.directory_generator.filenames)/batch_size,verbose=1)

		names = [int(n.split('/')[-1].replace('.jpg','')) for n in test_multilabel_datagen.directory_generator.filenames]

		photo_to_prediction_dict = dict(zip(names,predictions))

		print("Prediction part is finished.")

		prediction_to_df(photo_to_prediction_dict,photo_to_business_dict,x_biz).to_csv(test_data_root+"results.csv",index=False)
