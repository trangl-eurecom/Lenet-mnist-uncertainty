# This code is a application of LeNet-5 on MNIST
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.framework import dtypes
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import utils
import os
from dataset import DataSet

def process_mnist(images, dtype = dtypes.float32, reshape=True):
    if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

    return images

def get_data_info(images):
    rows, cols = images.shape
    std = np.zeros(cols)
    mean = np.zeros(cols)
    for col in range(cols):
        std[col] = np.std(images[:,col])
        mean[col] = np.mean(images[:,col])
    return mean, std

def standardize_data(images, means, stds):
    data = images.copy()
    rows, cols = data.shape
    for col in range(cols):
        if stds[col] == 0:
            data[:,col] = (data[:,col] - means[col])
        else:
            data[:,col] = (data[:,col] - means[col]) / stds[col]
    return data

def import_mnist():
    """
    This import mnist and saves the data as an object of our DataSet class
    :return:
    """
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 0
    ONE_HOT = True
    TRAIN_DIR = 'MNIST_data'


    local_file = base.maybe_download(TRAIN_IMAGES, TRAIN_DIR,
                                     SOURCE_URL + TRAIN_IMAGES)
    train_images = extract_images(open(local_file))

    local_file = base.maybe_download(TRAIN_LABELS, TRAIN_DIR,
                                     SOURCE_URL + TRAIN_LABELS)
    train_labels = extract_labels(open(local_file), one_hot=ONE_HOT)

    local_file = base.maybe_download(TEST_IMAGES, TRAIN_DIR,
                                     SOURCE_URL + TEST_IMAGES)
    test_images = extract_images(open(local_file))

    local_file = base.maybe_download(TEST_LABELS, TRAIN_DIR,
                                     SOURCE_URL + TEST_LABELS)
    test_labels = extract_labels(open(local_file), one_hot=ONE_HOT)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    ## Process images
    train_images = process_mnist(train_images)
    validation_images = process_mnist(validation_images)
    test_images = process_mnist(test_images)

    ## Standardize data
    train_mean, train_std = get_data_info(train_images)
#    train_images = standardize_data(train_images, train_mean, train_std)
#    validation_images = standardize_data(validation_images, train_mean, train_std)
#    test_images = standardize_data(test_images, train_mean, train_std)

    data = DataSet(train_images, train_labels)
    test = DataSet(test_images, test_labels)
    val = DataSet(validation_images, validation_labels)

    return data, test, val

# This function will create a train set (a subset of data) such that the number of each class are the same (if possible)
def extract_balance_train_set(data, data_size, train_size, nb_class):
    all_train_set = data.next_batch(data_size)
    allX = all_train_set[0]
    allY = all_train_set[1]
    nbEachClass = int(train_size / nb_class)
    cur_nbEachClass = np.zeros([nb_class])
    trainX = []
    trainY = []
    ind = 0
    while (len(trainX) < train_size) and (ind < data_size):
        class_nb = np.multiply(allY[ind], np.arange(nb_class))
        class_nb = int(np.sum(class_nb))
        if (cur_nbEachClass[class_nb] < nbEachClass):
            trainX.append(allX[ind])
            trainY.append(allY[ind])
            cur_nbEachClass[class_nb] = cur_nbEachClass[class_nb] + 1
        ind = ind + 1
        
    if (ind >= data_size):
        trainX = allX[0:train_size]
        trainY = allY[0:train_size]
        
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY
    
    
def logsumexp(vals, dim=None):
    m = tf.reduce_max(vals, dim)
    if dim is None:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - m), dim))
    else:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - tf.expand_dims(m, dim)), dim))
    
def log_cond_prob(output, latent_val):
    return tf.reduce_sum(output * latent_val, 2) - logsumexp(latent_val, 2)

if __name__ == '__main__':
    # Load MNIST
    FLAGS = utils.get_flags()
    data, test, _ = import_mnist()
    
    trainX, trainY = extract_balance_train_set(data, 55000, FLAGS.train_size, 10)
    testX = test.X
    testY = test.Y
    
    
    # Define training input, training output, validation input, 
    # validation output, test input and test output
    x = tf.placeholder(tf.float32, shape = [None, 784])
    y = tf.placeholder(tf.float32, shape = [None, 10])
    
    # DEFINE THE STUCTURE OF CNN
    # 01. convolution
    # x_4d is a tensor of [batch, 28, 28, 1]
    x_4d = tf.reshape(x, shape=[-1, 28, 28, 1])
    filters1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), tf.float32)
    bias1 = tf.Variable(tf.truncated_normal([1, 1, 1, 32], stddev=0.1), tf.float32)
    
    # conv1 is a tensor of [batch, 28, 28, 32]
    # --> height and width of conv1 layer 
    # --> padding algorithm SAME
    conv1 = tf.nn.conv2d(x_4d, filters1, strides=[1,1,1,1], padding = "SAME")
    
    # conv1_bias1: [batch, 28, 28, 32]
    conv1_bias1 = tf.add(conv1, bias1)
    
    # 02. Relu
    relu1 = tf.nn.relu(conv1_bias1)
    
    # 03. Max pooling: subsampling1: [batch, 14, 14, 32]
    subsampling1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    
    # 04. convolution: conv2_bias2: [batch, 14, 14, 64]
    filters2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1), tf.float32)
    bias2 = tf.Variable(tf.truncated_normal([1,1,1,64], stddev=0.1), tf.float32)
    conv2 = tf.nn.conv2d(subsampling1, filters2, strides=[1,1,1,1], padding="SAME")
    conv2_bias2 = tf.add(conv2, bias2)
    
    # 05. Relu
    relu2 = tf.nn.relu(conv2_bias2)
    # 06. Max pooling: subsampling2: [batch, 7, 7, 64]
    subsampling2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    
    # 07. Fully connected: conv3_bias3: [batch, 1, 1, 1024]
    filters3 = tf.Variable(tf.truncated_normal([7,7,64,1024], stddev=0.1), tf.float32)
    bias3 = tf.Variable(tf.truncated_normal([1,1,1,1024], stddev=0.1))
    conv3 = tf.nn.conv2d(subsampling2, filters3, strides=[1,7,7,1], padding = "VALID")
    conv3_bias3 = tf.add(conv3, bias3)
    
    # 08. Relu
    relu3 = tf.nn.relu(conv3_bias3)
    
    # 09. Fully connected conv4_bias4: [batch, 1, 1, 10]
    filters4 = tf.Variable(tf.truncated_normal([1,1,1024,10], stddev=0.1), tf.float32)
    bias4 = tf.Variable(tf.truncated_normal([1,1,1,10], stddev=0.1), tf.float32)
    conv4 = tf.nn.conv2d(relu3, filters4, strides=[1,1,1,1], padding="VALID")
    conv4_bias4 = tf.add(conv4, bias4)
    
    # 10. Calculate cross entropy between unscale outputs
    #     and labels (10-dim vector)
    predicted_vectors= tf.reshape(conv4_bias4, [-1, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_vectors, labels=y))
    
    # Compute the uncertainties and nll
    mc_test = 1;
    predicted_vectors_3d = tf.expand_dims(predicted_vectors, 0)
    uncertainties = tf.nn.softmax(predicted_vectors_3d, dim=-1)
    uncertainties = tf.reduce_mean(uncertainties, 0)
    
    nll = - tf.reduce_sum(-np.log(mc_test) + logsumexp(log_cond_prob(y, predicted_vectors_3d), 0))
    
    # TRAINING PHASE
    train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
    # COMPUTE THE ACCURACY OF CNN
    correct_vector = tf.equal(tf.argmax(predicted_vectors, 1), tf.argmax(y, 1))
    correct_vector = tf.cast(correct_vector, tf.float32)
    err = 1 - tf.reduce_mean(correct_vector)
    
    # RUN MODEL
    train_size = FLAGS.train_size
    batch_size = FLAGS.batch_size
    display_step = FLAGS.display_step
    n_iterations = FLAGS.n_iterations
    snapshot_step = int(n_iterations / 10)
    
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)
        prefix = str(train_size) + "mnist_LeNet5_" + "_snap_"
        filename = prefix + str(0)
        file = open(filename, 'w')
        cur_iter = open(str(train_size) + "_" + str(0), 'w')
        
        startID = 0
        endID = startID + batch_size
        for iteration in range(n_iterations):
            if (iteration > 0) and (iteration % 10 == 0):
                os.rename(str(train_size) + "_" + str(iteration - 10), str(train_size) + "_" + str(iteration))
            batch_X = trainX[startID:endID]
            batch_Y = trainY[startID:endID]
            startID = startID + batch_size
            endID = startID + batch_size
            if (endID > train_size):
                startID = 0
                endID = batch_size
            session.run(train, feed_dict={x: batch_X, y: batch_Y})
            
            if (iteration >= 0) and (iteration % display_step) == 0:
                [uncer, pred, negative_log_likelihood, err_rate] = session.run([uncertainties, predicted_vectors, nll, err], feed_dict={x: testX, y: testY})
                true_false = np.reshape((np.argmax(pred, 1) == np.argmax(testY, 1)), [len(pred), 1])
                uncer = np.concatenate((true_false, uncer), axis=-1)
                average_nll = negative_log_likelihood / len(testY)
                print("%d\t%g\t%g" % (iteration, err_rate, average_nll))
                file.write("%d\t%s\t%s\n" % (iteration, err_rate, average_nll))
                
            if (iteration >= 0) and (iteration % snapshot_step == 0):
                file.close()
                filename = prefix + str(iteration)
                file = open(filename, 'w')
                # Save the uncertainties
                np.savetxt(prefix + str(iteration) + "_uncer", uncer, fmt='%0.5f', delimiter='\t', newline='\n')