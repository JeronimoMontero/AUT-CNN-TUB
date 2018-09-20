# coding: utf-8

###########################################
#                                         #
#  source ~/envs/tensorflow/bin/activate  #
#                                         #
###########################################


import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import time
import datetime
import logging
import shutil
import boto3

from layers import conv_layer, max_pool_2x2, full_layer

# Create a resource service client, and select bucket
s3 = boto3.resource('s3')
bucket = s3.Bucket('imagesforcnn')

start = time.time()

home = os.getcwd().split('AUT-CNN-TUB')[0]
DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

########################################################################################################################
########################################################################################################################

ARCHITEKTURE = '3CONV_MEMORY'
MINIBATCH_SIZE = 50
STEPS = 15064
CONV1_DEPTH = 32

PIXEL = 28
COLOR = 3  # 3 or 1
CONV = 5
DROP = 1 - 0.2  # keep
LEARNING = 1e-3

# CFAK = 6
# CONV = int(round(PIXEL/CFAK, 0))

########################################################################################################################
########################################################################################################################

model_path = os.path.join(home,
                          'AUT-CNN-TUB', 'Data', 'Models',
                          'model_merge_{}_{}_b{}_s{}_{}'.format(ARCHITEKTURE, PIXEL, MINIBATCH_SIZE, STEPS, DATE))

if os.path.exists(model_path) is False:                   
    os.makedirs(model_path)
logger = logging.getLogger()

# create logger
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(model_path, 'model.log'))
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


if COLOR is 3:
    Gray = False
elif COLOR is 1:
    Gray = True
else:
    logger.error('COLOR has to be 1 or 3')

assert COLOR in [3, 1]

test_path = os.path.join(home, 'AUT-CNN-TUB/Data/TF_Images_final_{}/test/'.format(PIXEL))
train_path = os.path.join(home, 'AUT-CNN-TUB/Data/TF_Images_final_{}/train/'.format(PIXEL))
val_path = os.path.join(home, 'AUT-CNN-TUB/Data/TF_Images_final_{}/validate'.format(PIXEL))

test_number = len(os.listdir(test_path))
train_number = len(os.listdir(train_path))
val_number = len(os.listdir(val_path))

epoch = round((STEPS*MINIBATCH_SIZE)/train_number, 0)

logger.info(
    '''ARCHITEKTURE: {}, MINIBATCH_SIZE: {}, STEPS: {}, PIXEL: {}, COLOR: {}, CONV: {}, DROP_KEEP: {}, EPOCH: {}, 
    LEARNING: {}, CONV1_DEPTH: {}'''.format(ARCHITEKTURE, MINIBATCH_SIZE,
                                          STEPS, PIXEL, COLOR, CONV, DROP,
                                          epoch, LEARNING, CONV1_DEPTH))

labels = ['01.0', '02.0', '03.0', '04.0', '05.0', '06.0', '07.1', '07.2',
          '08.0', '09.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0']

position_dict = {k: v for v, k in enumerate(labels)}
label_dict = {v: k for v, k in enumerate(labels)}


def label_to_binary(position_dict, label):
    z = np.zeros(len(position_dict), dtype=int)
    z[position_dict[label]] = 1
    return z


class DataGetter:
    def __init__(self, path, GRBtoGray=False):
        self.path = path
        self.GRBtoGray = GRBtoGray
        self.batchindices = None
        self.data_path = os.path.join(self.path, '*g') 
        self.files = glob.glob(self.data_path)
        self.num_img = len(self.files)
        self.Data = list(range(0, self.num_img))

        img_list = []
        label_list = []
        label_bin_list = []

        for file_path in self.files:

            img = cv2.imread(file_path)
            label_str = file_path.split('/')[-1][:4]
            label_bin = label_to_binary(position_dict, label_str)

            img_list.append(img)
            label_bin_list.append(label_bin)
            label_list.append(file_path.split('/')[-1])

        self.img_list = np.array(img_list)
        self.label_list = np.array(label_list)
        self.label_bin = np.array(label_bin_list)

    def get_batch(self, size):
        
        if len(self.Data) > size:
            self.batchindices = list(np.random.choice(self.Data, size=size, replace=False))
            for x in self.batchindices:
                self.Data.remove(x)

        else:  # new epoch
            self.batchindices = np.random.choice(self.Data, size=len(self.Data), replace=False)  
            self.Data = list(range(0, self.num_img))

        return (self.img_list[self.batchindices],
                self.label_bin[self.batchindices],
                self.label_list[self.batchindices].tolist())


test_img = DataGetter(test_path, Gray)
train_img = DataGetter(train_path, Gray)
val_img = DataGetter(val_path, Gray)


x = tf.placeholder(tf.float32, shape=[None, PIXEL, PIXEL, COLOR])
y_ = tf.placeholder(tf.float32, shape=[None, len(position_dict)])


conv1 = conv_layer(x, shape=[CONV, CONV, COLOR, CONV1_DEPTH])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[CONV, CONV, CONV1_DEPTH, CONV1_DEPTH * 2])
conv2_pool = max_pool_2x2(conv2)

conv3 = conv_layer(conv2_pool, shape=[CONV, CONV, CONV1_DEPTH * 2, CONV1_DEPTH * 4])

new_size = int(PIXEL/4*PIXEL/4 * CONV1_DEPTH * 4)

if PIXEL%2 is not 0:
    logger.warning('potential issue with the pixel size')	

conv3_flat = tf.reshape(conv3, [-1, new_size])
full_1 = tf.nn.relu(full_layer(conv3_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, len(position_dict))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(LEARNING).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epoch_counter = 0 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):

        X_batch, y_batch, _ = train_img.get_batch(MINIBATCH_SIZE)

        if i % int(STEPS/100) == 0:

            #########
            # train
            #########

            train_accuracy = sess.run(accuracy, feed_dict={x: X_batch, y_: y_batch, keep_prob: 1.0})
            cross_entro = sess.run(cross_entropy, feed_dict={x: X_batch, y_: y_batch, keep_prob: 1.0})

            ##############
            # validation
            ##############
            val_ac_list = []
            val_entro_list = []

            val_batch = val_number

            val_counter = 0
            while val_counter < val_number//val_batch:
                val_counter += 1

                vX_batch, vy_batch, file_name = val_img.get_batch(val_batch)
                val_accuracy = sess.run(accuracy, feed_dict={x: vX_batch, y_: vy_batch, keep_prob: 1.0})
                cross_entro_val = sess.run(cross_entropy, feed_dict={x: vX_batch, y_: vy_batch, keep_prob: 1.0})
                val_ac_list.append(val_accuracy)
                val_entro_list.append(cross_entro_val)

            if val_number % val_batch is not 0:

                vX_batch, vy_batch, file_name = val_img.get_batch(val_batch)
                val_accuracy = sess.run(accuracy, feed_dict={x: vX_batch, y_: vy_batch, keep_prob: 1.0})
                cross_entro_val = sess.run(cross_entropy, feed_dict={x: vX_batch, y_: vy_batch, keep_prob: 1.0})
                val_ac_list.append(val_accuracy * (val_number % val_batch) / val_batch)
                val_entro_list.append(cross_entro_val * (val_number % val_batch) / val_batch)

            logger.info(
                "step {}, validation accuracy {}".format(i, (np.sum(val_ac_list) / val_number) * val_batch))
            logger.info(
                "step {}, validation cross entropy {}".format(i, (np.sum(val_entro_list) / val_number) * val_batch))

            logger.info(
                "step {}, train accuracy {}".format(i, np.mean(train_accuracy)))
            logger.info(
                "step {}, train cross entropy {}".format(i, np.mean(cross_entro)))

        sess.run(train_step, feed_dict={x: X_batch, y_: y_batch, keep_prob: DROP})

    #########
    # test
    #########

    test_ac_list = []

    test_batch = 4000

    test_counter = 0
    while test_counter < test_number//test_batch:
        test_counter += 1

        tX_batch, ty_batch, file_name = test_img.get_batch(test_batch) 
        test_accuracy = sess.run(accuracy, feed_dict={x: tX_batch, y_: ty_batch, keep_prob: 1.0})
        test_ac_list.append(test_accuracy)

    if test_number % test_batch is not 0:

        tX_batch, ty_batch, file_name = test_img.get_batch(test_batch)
        test_accuracy = sess.run(accuracy, feed_dict={x: tX_batch, y_: ty_batch, keep_prob: 1.0})
        test_ac_list.append(test_accuracy * (test_number % test_batch) / test_batch)

    logger.info("test accuracy: {}".format((np.sum(test_ac_list) / test_number) * test_batch))

    model_path_file = os.path.join(model_path, 'model_merge_{}_b{}_s{}_{}.ckpt'.format(
                                PIXEL, MINIBATCH_SIZE, STEPS, DATE))
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_path_file)

logger.info("Model saved in path: %s" % save_path)

end = time.time()

logger.info('Model training took, {} seconds'.format(str(end-start)))


# compress
shutil.make_archive(model_path,
                    'zip',
                    model_path
                    )

# Upload a new file
path = model_path + '.zip'
data = open(os.path.join(home, path), 'rb')
folder = 'models'
subfolder = None
name = path.split('/')[-1]

if folder is not None:
    if subfolder is not None:
        key = '{}/{}/{}'.format(folder, subfolder, name)
    else:
        key = '{}/{}'.format(folder, name)
else: 
    key = '{}'.format(name)

s3.Bucket('imagesforcnn').put_object(Key=key, Body=data)

logger.info('Model saved on S3')


