from keras.datasets import cifar10
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not os.path.exists('model'):
    os.makedirs('model')



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()

X_train=X_train/255.0
X_test=X_test/255.0
from keras.utils import np_utils

Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)

import tensorflow as tf
X = tf.placeholder("float",[None,32,32,3]) # as our dataset
Y = tf.placeholder("float",[None,10])
y_true_cls = tf.argmax(Y, dimension=1)
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

W_C1 = init_weights([5,5,3,32])# 3x3x3 conv, 32 outputs
W_C2 = init_weights([3,3,32,64])# 3x3x32 conv, 64 outputs
W_C3 = init_weights([3, 3, 64, 128])# 3x3x64 conv, 128 outputs

W_FC = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
W_O = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)


p_keep_conv = tf.placeholder("float") #for dropouts as percentage
p_keep_hidden = tf.placeholder("float")

def model(X, W_C1, W_C2, W_C3, W_FC, W_O, p_keep_conv,p_keep_hidden):

    C1 = tf.nn.relu(tf.nn.conv2d(X,W_C1,
                                strides=[1,1,1,1], padding = "SAME"))

    P1 = tf.nn.max_pool(C1,ksize=[1,2,2,1],
                         strides=[1,2,2,1], padding = "SAME" ) # 1st pooling layer shape =(?,14,14,32)

    D1 = tf.nn.dropout(P1,p_keep_conv) # 1st dropout at conv

    C2 = tf.nn.relu(tf.nn.conv2d(D1,W_C2,
                                strides=[1,1,1,1], padding = "SAME")) # 2nd convoultion layer shape=(?, 14, 14, 62)

    P2 = tf.nn.max_pool(C2,ksize=[1,2,2,1],
                         strides=[1,2,2,1], padding = "SAME" ) # 2nd pooling layer shape =(?,7,7,64)

    D2 = tf.nn.dropout(P2,p_keep_conv) # 2nd dropout at conv


    C3 = tf.nn.relu(tf.nn.conv2d(D2,W_C3,
                                strides=[1,1,1,1], padding = "SAME")) # 3rd convoultion layer shape=(?, 7, 7, 128)

    P3 = tf.nn.max_pool(C3,ksize=[1,2,2,1],
                         strides=[1,2,2,1], padding = "SAME" ) # 3rd pooling layer shape =(?,4,4,128)

    P3 = tf.reshape(P3, [-1, W_FC.get_shape().as_list()[0]])    # reshape to (?, 2048)
    D3 = tf.nn.dropout(P3, p_keep_conv) # 3rd dropout at conv

    FC = tf.nn.relu(tf.matmul(D3,W_FC))
    FC = tf.nn.dropout(FC, p_keep_hidden) #droput at fc

    output = tf.matmul(FC,W_O)

    return output

get_m = model(X, W_C1, W_C2, W_C3, W_FC, W_O, p_keep_conv,p_keep_hidden)

Y_pred = tf.nn.softmax(get_m ,name='Y_pred')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = get_m ,labels = Y))
y_pred_cls = tf.argmax(Y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=get_m, labels=Y)
optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(get_m, 1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
X_train = X_train.reshape(-1,32,32,3)
X_test = X_test.reshape(-1,32,32,3)
saver = tf.train.Saver()
def train():

    print("Loop\t\tTraining Loss\t\t\tTraining Accuracy\t\t\tTesting Loss\t\t\tTesting Accuray")
    from math import ceil, floor
    def float_round(num, places = 0, direction = floor):
        return direction(num * (10**places)) / float(10**places)
    epochs = 500
    save_path = './model/trained_model'
    import numpy as np
    with tf.Session() as sess:
        # you need to initialize all variables
        sess.run(tf.global_variables_initializer())
        feed_dict_val = {X: X_test, Y: Y_test,p_keep_conv: 1.0, p_keep_hidden: 1.0}
        feed_dictr = {}
        for epoch in range(epochs):

            for start, end in zip(range(0, len(X_train), 128), range(128, len(X_train)+1, 128)):
                feed_dictr={X : X_train[start:end] , Y : Y_train[start:end], p_keep_conv: 0.25, p_keep_hidden: 0.5}
                sess.run(optimizer,feed_dictr)

            if epoch % 50 == 0:
                val_loss = sess.run(cost, feed_dict=feed_dict_val)
                _, l, predictions = sess.run([optimizer, cost, Y_pred], feed_dict=feed_dictr)
                acc = sess.run(accuracy, feed_dict=feed_dictr)
                val_acc = sess.run(accuracy, feed_dict=feed_dict_val)
                print("{}/{}\t\t{}\t\t\t\t\t{}\t\t\t\t{}\t\t\t\t{}".format(epoch,epochs,float_round(l, 4),float_round(acc*100, 4),float_round(val_loss, 4) ,float_round(val_acc*100, 4)))
                saver.save(sess, save_path)
    print("Model saved in file: {}".format(save_path))

import  sys

import matplotlib.pyplot as plt
def plot_conv_layer(layer, image):
    import math
    import matplotlib.pyplot as plt
    # Assume layer_output is a 4-dim tensor
    # e.g. output_conv1 or output_conv2.

    # Create a feed-dict which holds the single input image.
    # Note that TensorFlow needs a list of images,
    # so we just create a list with this one image.
    feed_dict = {X: [image]}
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Retrieve the output of the layer after inputting this image.
    values = sess.run(layer, feed_dict=feed_dict)

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i<num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.savefig('CONV_rslt.png')
    plt.show()
	

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
    sess.run(tf.global_variables_initializer())
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('model/trained_model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('model'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()
    '''
    f = open('helloworld.txt','w')
    s = [n.name for n in tf.get_default_graph().as_graph_def().node]	
    for item in s:
        f.write("%s\n" % item)
    '''

    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("Y_pred:0")

    ## feed the images to the input placeholders
    x = graph.get_tensor_by_name("Placeholder:0")
    y_true = graph.get_tensor_by_name("Placeholder_1:0")
    Pl = graph.get_tensor_by_name("Placeholder_2:0")
    Pl_1 = graph.get_tensor_by_name("Placeholder_3:0")
    layer  = graph.get_tensor_by_name("Conv2D:0")
    y_test_images = np.zeros((1, 10))

    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images, Pl: 1.0,Pl_1: 1.0}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_class, .....]
    import numpy as np
    v = np.amax(result)
    a, b = np.where(result == v)
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(classes[int(b)])
    plot_conv_layer(layer,image )


import sys
def main(argv):

    if argv[0] in ["predict", "test"]:
        #value = argv[0]
        test(argv[1])

    if argv[0] == "train":
        #value = argv[0]
        train()

if __name__ == "__main__":
    main(sys.argv[1:])
