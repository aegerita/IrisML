import os

import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" check version """
# print("TensorFlow version: {}".format(tf.__version__))
# print("Eager execution: {}".format(tf.executing_eagerly()))

""" download dataset """
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
# print("Local copy of the dataset file: {}".format(train_dataset_fp))

""" label the column name of the CSV file """
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]  # python seems very interesting! it includes everything to the last one!
label_name = column_names[-1]
class_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
# print("Features: {}".format(feature_names))
# print("Label: {}".format(label_name))

""" reformat for the training model """
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    32,  # batch size
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
feature, label = next(iter(train_dataset))


# print(features)


def pack_features_vector(feature, label):
    """ Pack the features into a single array """
    feature = tf.stack(list(feature.values()), axis=1)
    return feature, label


train_dataset = train_dataset.map(pack_features_vector)
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
print(tf.nn.softmax(predictions))
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
