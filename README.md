
# DL4CV - Group 55
This project is done for Deep Learning for Computer Vision course (WiSe 2017/18) offered by Technical University of Munich, Faculty of Informatics, Computer Vision Group.
Our project consist of 2 parts. 
1. Using CNNs (convolutional neural networks) to predict tags for restaurant images. 
This is a kaggle competition which is organized by Yelp.
https://www.kaggle.com/c/yelp-restaurant-photo-classification
2. Using RNNs (Recurrent Neural Networks) to generate fake reviews for restaurants.
The dataset can be downloaded via https://www.yelp.com/dataset.

## Dependencies

In this project, we have used ```Keras``` with ```Tensorflow``` backend. We have installed these dependencies via ```anaconda```. We have used pandas for dataframe operations.

- keras 2.1.3
- tensorflow 1.4.1
- pandas 0.20.3

## Folder Structure

The folder structure should be like below:

/split.py  
/organize_photos.py  
/reorganize_photos.py  
/MultilabelGenerator.py  
/TopClassifier.py  
/utility.py  
/VGG16.py  
/VGG16_bottleneck.py  
/InceptionV3model.py   
/InceptionV3_bottleneck.py  
/XceptionModel.py  
/Xception_bottleneck.py  
/train_vgg16.py  
/extract_vgg16_bottleneck_features.py  
/train_vgg16_bottleneck.py  
/train_inceptionv3.py  
/extract_inceptionV3_bottleneck_features.py  
/train_inceptionV3_bottleneck.py  
/train_xception.py  
/extract_xception_bottleneck_features.py  
/train_xception_bottleneck.py  
/predict_inceptionv3.py  
/predict_xception.py  
/predict_test.py   
/parse_reviews.py   
/RNNTextGeneration.py   
/train_rnn.py  
/RNN.py   
/generate_text.py   
/data/  
/data/learning/  
/data/learning/review.json  
/data/learning/train.csv  
/data/learning/train_photo_to_biz_ids.csv  
/data/learning/photos - all training photos should be in this folder first  
/data/learning/photos/train  
/data/learning/photos/validation  
/data/learning/models - it should be empty before training  
/data/test  
/data/test/test_photo_to_biz.csv  
/data/test/photos   
/data/test/photos/test - all test photos should be in this folder  

## Training

### 1. CNN

#### Preparing the data for Hyperparameter Research
1. Check that all ```train.csv``` and ```train_photo_to_biz_ids.csv``` are in the ```/data/learning/``` folder, and all of the training photos are in the ```/data/learning/photos/``` folder.
2. Run ```python3 split.py``` to split the learning data to training and validation. Don't forget to set the ```test_size``` attribute in ```line 5``` to your wanted rate of training and validation data.
3. Run ```python3 organize_photos.py``` to move the training and validation photos to their folders.

#### Preparing the data for Training
1. Check that all ```train.csv``` and ```train_photo_to_biz_ids.csv``` are in the ```/data/learning/``` folder, and all of the training photos are in the ```/data/learning/photos/``` folder.
2. Run ```python3 reorganize_photos.py``` command to move the photos their original folder.
3. Run ```python3 split.py``` to split the learning data to training and validation. Don't forget to set the ```test_size``` attribute in ```line 5``` to ```0``` for no validation data.
4. Run ```python3 organize_photos.py``` to move the training and validation photos to their folders.

#### Training InceptionV3
1. Run ```python3 train_inceptionv3.py``` command. After this command executes, there should be a ```weights_inceptionv3.hdf5``` file in your ```/data/learning/models/``` folder.
2. Run ```python3 predict_inceptionv3.py``` command to get the predictions and F1-Scores for your training (and for hyperparameter research validation) data.

#### Training Xception
1. Run ```python3 train_xception.py``` command. After this command executes, there should be a ```weights_xception.hdf5``` file in your ```/data/learning/models/``` folder.
2. Run ```python3 predict_xception.py``` command to get the predictions and F1-Scores for your training (and for hyperparameter research validation) data.

#### Test Data
1. Run ```python3 predict_test.py``` command. After this command executes, there should be a ```results.csv``` file in your ```/data/test/``` folder. You can upload this file to kaggle.

### 2. RNN

#### Preparing the data for RNNs
1. Check that your ```review.json``` file is in the folder ```/data/learning/```.
2. Run ```python3 parse_reviews.py``` command, to generate a better file to feed RNN.

#### Training RNN
1. Run ```python3 train_rnn.py``` command. This command will create ```rnn.hdf5``` file under ```/data/learning/models/``` folder.

#### Generating Text via RNN
1. Run ```python3 generate_text.py``` command. This command will generate text for each business label.


## Sources

#### Keras Transfer Learning & Image Preprocessing
https://keras.io/applications
https://keras.io/preprocessing/image/
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://www.youtube.com/watch?v=BhQW2OLzx_c
https://medium.com/@chengweizhang2012/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model-9b0f6b4c1b0d
https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb

#### Multi-label classification
https://stackoverflow.com/questions/44164749/how-does-keras-handle-multilabel-classification
http://www.kubacieslik.com/extending-keras-imagedatagenerator-handle-multilable-classification-tasks/
https://depends-on-the-definition.com/guide-to-multi-label-multi-class-classification-with-neural-networks-in-python/

#### RNNs
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/


