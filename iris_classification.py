import os

import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" check version """
print("TensorFlow version: {}".format(tf.__version__))
# print("Eager execution: {}".format(tf.executing_eagerly()))


""" download dataset (training and testing) """
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
# print("Local copy of the dataset file: {}".format(train_dataset_fp))
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

""" label the column name of the CSV file """
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]  # python seems very interesting! it includes everything to the last one!
label_name = column_names[-1]
class_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
# print("Features: {}".format(feature_names))
# print("Label: {}".format(label_name))


""" reformat for the training model """
batch_size = 32
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
feature, label = next(iter(train_dataset))

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)


# print(features)


def pack_features_vector(the_feature, the_label):
    """ Pack the features into a single array """
    the_feature = tf.stack(list(the_feature.values()), axis=1)
    return the_feature, the_label


train_dataset = train_dataset.map(pack_features_vector)
test_dataset = test_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))
# print(features[:5])


""" making the model: 2 layer, 10 nodes each """
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

""" output the prediction """
predictions = model(features)
# print(tf.nn.softmax(predictions))
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))


def plot(x, y):
    """ plot the scatter plot """
    plt.scatter(feature[column_names[x]],
                feature[column_names[y]],
                c=label,
                cmap='viridis')
    plt.xlabel("{}".format(column_names[x]))
    plt.ylabel("{}".format(column_names[y]))
    plt.show()


# plot(2, 0)


""" calculate loss """
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(training_model, x, y, training):
    """ calculate loss """
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = training_model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


# loss = loss(model, features, labels, training=False)
# print("Loss test: {}".format(loss))


def grad(model, inputs, targets):
    """ find gradient """
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


""" a single optimization step"""
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
lost, grads = grad(model, features, labels)
print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(),
                                  lost.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(),
                                  loss(model, features, labels, training=True).numpy()))

# Note: Rerunning this cell uses the same model variables
""" keeps training and optimization """
# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y, model(x, training=True))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 10 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

""" plot the loss in training """
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

""" evaluate accuracy of the model """
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    logits = model(x, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

# print(tf.stack([y, prediction], axis=1))
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

""" apply the model on examples """
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5, ],
    [5.9, 3.0, 4.2, 1.5, ],
    [6.9, 3.1, 5.4, 2.1]
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))

# Save the entire model as a SavedModel.
model.save('iris_model')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model('iris_model')
tflite_model = converter.convert()
open("iris_model.tflite", "wb").write(tflite_model)
