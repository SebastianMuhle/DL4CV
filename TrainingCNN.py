import tensorflow as tf
from .InceptionV3model import InceptionV3model
from .XceptionModel import XceptionModel

# Define input function
def imgs_input_fn(filenames, labels=None, perform_shuffle=None, repeat_count=1, batch_size=1):
    # Labels=None for interferences. If label=True and training also shuffle the data!!!
    batch_features, batch_labels = 1,1
    return batch_features, batch_labels

# Test input function
next_batch = imgs_input_fn(test_files, labels=test_labels, perform_shuffle=True, batch_size=20)
with tf.Session() as sess:
    first_batch = sess.run(next_batch)
x_d = first_batch[0]['input_1']

print(x_d.shape)
img = image.array_to_img(x_d[8])
img.show()

# Do to: Do following in a grid search  for hyper parameter tuning

# Hyperparameters
num_freezedLayers=249

# Create optimizer
optimizer = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9)

# Create model
model = InceptionV3model().create_model(num_freezedLayers=num_freezedLayers, optimizer=optimizer)
estimator_model = tf.keras.estimator.model_to_estimator(keras_model= model)

# Training
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

tf.estimator.train_and_evaluate(estimator_model, train_spec, eval_spec)