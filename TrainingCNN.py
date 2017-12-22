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
num_freezedLayersArray=[5, 80, 249]
learning_rates=[0.01, 0.001, 0.0001]

# Hyperparameter search
for num_freezedLayers in num_freezedLayersArray:
    for lr in learning_rates:

        # Create Optimizer
        optimizerSGD = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9)
        optimizerAdam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,)

        # Create model
        model = InceptionV3model().create_model(num_freezedLayers=num_freezedLayers, optimizer=optimizerSGD)
        estimator_model = tf.keras.estimator.model_to_estimator(keras_model= model)

        # Training
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(test_files,
                                                                           labels=test_labels,
                                                                           perform_shuffle=True,
                                                                           repeat_count=5,
                                                                           batch_size=20), max_steps=500)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(test_files,
                                                                         labels=test_labels,
                                                                         perform_shuffle=False,
                                                                         batch_size=1))

        tf.estimator.train_and_evaluate(estimator_model, train_spec, eval_spec)