from keras.datasets import cifar10
import numpy as np
import os

##create directpry model if it does not exist
if not os.path.exists('model'):
    os.makedirs('model')



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

#load data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


X_train = X_train / 255.0
X_test = X_test / 255.0

from keras.utils import np_utils

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

import tensorflow as tf
session = tf.Session()
tf.logging.set_verbosity(tf.logging.ERROR)


X = tf.placeholder(tf.float32, shape = [None, 32, 32, 3], name = 'X')  # as our dataset 
Y = tf.placeholder(tf.float32, shape = [None, 10], name = 'Y')#


y_true_cls = tf.argmax(Y, dimension=1)
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    ## weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## biases using the create_biases function
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    ## max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    ## Output of pooling is fed to Relu
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):

    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=X,
                                         num_input_channels=3,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=10,
                            use_relu=False)

Y_pred = tf.nn.softmax(layer_fc2, name='Y_pred')
y_pred_cls = tf.argmax(Y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=Y)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Y_pred ,labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer())

X_train = X_train.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)

from math import ceil, floor
def float_round(num, places = 0, direction = floor):
    return direction(num * (10**places)) / float(10**places)

def show_progress(n, count, feed_dict_train, feed_dict_validate, val_loss, l):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    print("{}/{}\t\t{}\t\t\t\t\t{}\t\t\t\t{}\t\t\t\t{}".format(count,n,float_round(l, 4),float_round(acc*100, 4),float_round(val_loss, 4) ,float_round(val_acc*100, 4)))


saver = tf.train.Saver()


def train():
    feed_dict_tr = {}
    feed_dict_val = {X: X_test, Y: Y_test}
    epochs = 100
    num = int(epochs/10)
    save_path = './model/trained_model'
    count = 0

    import numpy as np

    print("Loop\t\tTraining Loss\t\t\tTraining Accuracy\t\t\tTesting Loss\t\t\tTesting Accuray")
    with tf.Session() as sess:
        # initialize all variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for start, end in zip(range(0, len(X_train), 128), range(128, len(X_train) + 1, 128)):
                feed_dict_tr = {X: X_train[start:end], Y: Y_train[start:end]}
                session.run(optimizer, feed_dict=feed_dict_tr)

            if epoch % 10 == 0:
                count = count+1
                val_loss = session.run(cost, feed_dict=feed_dict_val)
                _, l, predictions = session.run([optimizer, cost, Y_pred], feed_dict=feed_dict_tr)
                show_progress(num, count, feed_dict_tr, feed_dict_val, val_loss, l)
                saver.save(session, save_path)

    print("Model saved in file: {}".format(save_path))


import  sys

def test(image):
    import tensorflow as tf
    import numpy as np
    import os, glob, cv2
    import sys, argparse

    # First, pass the path of the image
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = image
    filename = dir_path + '/' + image_path
    image_size = 32
    num_channels = 3
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    ##restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('.\\model\\trained_model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('.\\model'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("Y_pred:0")

    ## feed the images to the input placeholders
    x = graph.get_tensor_by_name("X:0")
    y_true = graph.get_tensor_by_name("Y:0")
    y_test_images = np.zeros((1, 10))

    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_class, .....]
    import numpy as np
    v = np.amax(result)
    a, b = np.where(result == v)
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(classes[int(b)])

def main(argv):

    if argv[0] in ["predict", "test"]:
        test(argv[1])

    if argv[0] == "train":
        train()

if __name__ == "__main__":
    main(sys.argv[1:])
