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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# load helper functions for the net architecture
from layers import conv_layer, max_pool_2x2, full_layer

sns.set_style("whitegrid")


# Create a resource service client, and select bucket
s3 = boto3.resource('s3')
bucket = s3.Bucket('imagesforcnn')

start = time.time()

home = os.getcwd().split('AUT-CNN-TUB')[0]
DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

########################################################################################################################
########################################################################################################################

###############
# set parameter
###############

ARCHITECTURE = '3CONV_MEMORY'  # tag
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
                          'model_merge_{}_{}_b{}_s{}_{}'.format(ARCHITECTURE, PIXEL, MINIBATCH_SIZE, STEPS, DATE))

if os.path.exists(model_path) is False:
    os.makedirs(model_path)
logger = logging.getLogger()

# create logger
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(model_path, 'model.log'))
fh.setLevel(logging.DEBUG)

# create console handler with same log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# check chanel
if COLOR is 3:
    Gray = False
elif COLOR is 1:
    Gray = True
else:
    logger.error('COLOR has to be 1 or 3')
#
assert COLOR in [3, 1]

# path to the data set
test_path = os.path.join(home, 'AUT-CNN-TUB/Data/TF_Images_final_{}/test/'.format(PIXEL))
train_path = os.path.join(home, 'AUT-CNN-TUB/Data/TF_Images_final_{}/train/'.format(PIXEL))
val_path = os.path.join(home, 'AUT-CNN-TUB/Data/TF_Images_final_{}/validate'.format(PIXEL))

# get number of images
test_number = len(os.listdir(test_path))
train_number = len(os.listdir(train_path))
val_number = len(os.listdir(val_path))

# get number of epochs
epoch = round((STEPS*MINIBATCH_SIZE)/train_number, 0)

# logging parameters
logger.info('ARCHITECTURE: {}, MINIBATCH_SIZE: {}, STEPS: {}, PIXEL: {}, COLOR: {}, CONV: {}, DROP_KEEP: {}, '
            'EPOCH: {}, LEARNING: {}, CONV1_DEPTH: {}'.format(ARCHITECTURE, MINIBATCH_SIZE,
                                                              STEPS, PIXEL, COLOR, CONV, DROP,
                                                              epoch, LEARNING, CONV1_DEPTH))

# labeling
labels = ['01.0', '02.0', '03.0', '04.0', '05.0', '06.0', '07.1', '07.2',
          '08.0', '09.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0']

position_dict = {k: v for v, k in enumerate(labels)}
label_dict = {v: k for v, k in enumerate(labels)}


def label_to_binary(position_dict , label):
    """
    translate label in one-hot tensor like 03.0 -> [0, 0, 0, 3, 0, ... , 0]

    :param position_dict:
    :param label:
    :return: one-hot tensor
    """
    z = np.zeros(len(position_dict), dtype=int)
    z[position_dict[label]] = 1
    return z


class DataGetter:
    """
    DataGetter loads all images from the given file in  memory and stores it there for fast access.
    With the get_batch function it is possible to randomly draw, the given batch-size without return.
    """
    def __init__(self, path, GRBtoGray=False):
        """
        :param path: path to the image folder, eg. ../test/
        :param GRBtoGray: flag for loading the pictures in grayscale if True
        """

        self.path = path
        self.GRBtoGray = GRBtoGray
        self.batch_indices = None
        self.data_path = os.path.join(self.path, '*g')
        self.files = glob.glob(self.data_path)
        self.num_img = len(self.files)
        self.Data = list(range(0, self.num_img))

        img_list = []
        label_list = []
        label_bin_list = []

        for file_path in self.files:

            if self.GRBtoGray is True:
                img = cv2.imread(file_path, flags=0)
                img = img.reshape(img.shape[0], img.shape[1], 1)
            else:
                img = cv2.imread(file_path, flags=1)
            label_str = file_path.split('/')[-1][:4]
            label_bin = label_to_binary(position_dict, label_str)

            img_list.append(img)
            label_bin_list.append(label_bin)
            label_list.append(file_path.split('/')[-1])

        self.img_list = np.array(img_list)
        self.label_list = np.array(label_list)
        self.label_bin = np.array(label_bin_list)

    def get_batch(self, size):
        """
        :param size: patch size
        :return: (array(images), array(on hot labels), list[file names])
        """
        if len(self.Data) > size:
            self.batch_indices = list(np.random.choice(self.Data, size=size, replace=False))
            for x in self.batch_indices:
                self.Data.remove(x)

        else:  # new epoch
            self.batch_indices = np.random.choice(self.Data, size=len(self.Data), replace=False)
            self.Data = list(range(0, self.num_img))

        return (self.img_list[self.batch_indices],
                self.label_bin[self.batch_indices],
                self.label_list[self.batch_indices].tolist())


# initialise DataGetter object
test_img = DataGetter(test_path, Gray)
train_img = DataGetter(train_path, Gray)
val_img = DataGetter(val_path, Gray)

########################################################################################################################
# tensorflow network architecture
########################################################################################################################

# placeholder for images and labels
x = tf.placeholder(tf.float32, shape=[None, PIXEL, PIXEL, COLOR])
y_ = tf.placeholder(tf.float32, shape=[None, len(position_dict)])

# 1 convolution layer with max pooling
conv1 = conv_layer(x, shape=[CONV, CONV, COLOR, CONV1_DEPTH])
conv1_pool = max_pool_2x2(conv1)

# 2 convolution layer with max pooling
conv2 = conv_layer(conv1_pool, shape=[CONV, CONV, CONV1_DEPTH, CONV1_DEPTH * 2])
conv2_pool = max_pool_2x2(conv2)

new_size = int(PIXEL/4*PIXEL/4 * CONV1_DEPTH * 2)

if PIXEL % 2 is not 0:
    logger.warning('potential issue with the pixel size')

# fully connected flatt layer
conv3_flat = tf.reshape(conv2_pool, [-1, new_size])
full_1 = tf.nn.relu(full_layer(conv3_flat, 1024))

# drop out
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

# output layer
y_conv = full_layer(full1_drop, len(position_dict))

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))

# optimizer
train_step = tf.train.AdamOptimizer(LEARNING).minimize(cross_entropy)

# predictor
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

########################################################################################################################
# tensorflow session
########################################################################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        # get train batch
        X_batch, y_batch, _ = train_img.get_batch(MINIBATCH_SIZE)

        # calculate outcome for logging
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
    # save the model
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_path_file)

########################################################################################################################

logger.info("Model saved in path: %s" % save_path)

end = time.time()

logger.info('Model training took, {} seconds'.format(str(end - start)))

# make graphs
with open((os.path.join(model_path, 'model.log'))) as f:
    f = f.readlines()

steps_entro_train = []
steps_ac_train = []
steps_entro_val = []
steps_ac_val = []
step = []
epoch = []
epoch_entro_test = []
epoch_ac_test = []
epoch_ac_train = []

for line in f:
    if 'step' in line:
        if 'validation cross entropy' in line:
            step.append(int(line.split('step')[1].split(',')[0]))
            steps_entro_val.append(float(line.split('validation cross entropy ')[-1][:-1]))

        elif 'validation accuracy' in line:
            steps_ac_val.append(float(line.split('validation accuracy ')[-1][:-1]))

        elif 'train accuracy' in line:
            steps_ac_train.append(float(line.split('train accuracy ')[-1][:-1]))
        elif 'train cross entropy' in line:
            steps_entro_train.append(float(line.split('train cross entropy ')[-1][:-1]))

    elif 'epoch' in line:
        if 'test accuracy' in line:
            epoch.append(int(line.split('epoch')[1].split(',')[0]))
            epoch_ac_test.append(float(line.split('test accuracy ')[1][:-1]))
        elif 'training accuracy' in line:
            epoch_ac_train.append(float(line.split('training accuracy ')[1][:-1]))
        elif 'cross' in line:
            epoch_entro_test.append(float(line.split('cross entropy ')[-1][:-1]))

fig, ax1 = plt.subplots(figsize=(15, 10))

ax1.plot(step, steps_entro_val, 'bo', label='Val.')
ax1.plot(step, steps_entro_val, 'b', lw=2.5)
ax1.plot(step, steps_entro_train, 'yo', label='Train')
ax1.plot(step, steps_entro_train, 'y', lw=2.5)
ax1.set_ylim([0, 20])
ax1.set_yticks(list(range(0, 21, 4)))
ax1.set_xlabel('Steps').set_fontsize(20)
ax1.tick_params('x', labelsize=15)
ax1.legend(loc=6, fontsize=15)
ax1.set_ylabel('Cross Entropy', color='b').set_fontsize(20)
ax1.tick_params('y', colors='b', labelsize=15)

ax2 = ax1.twinx()
ax2.set_ylim([0, 1])
ax2.plot(step, steps_ac_val, 'ro', label='Val.')
ax2.plot(step, steps_ac_val, 'r', lw=2.5)
ax2.plot(step, steps_ac_train, 'go', label='Train')
ax2.plot(step, steps_ac_train, 'g', lw=2.5)
ax2.set_ylabel('Accuracy', color='r').set_fontsize(20)
ax2.tick_params('y', colors='r', labelsize=15)
ax2.legend(loc=7, fontsize=15)
fig.tight_layout()
plt.savefig(model_path + '/graph')
plt.close()

value_list = []
parameter_list = ['ARCHITECTURE', 'MINIBATCH_SIZE', 'STEPS', 'PIXEL', 'COLOR', 'CONV',
                  'EPOCH', 'CONV1_DEPTH']

for parameter in parameter_list:
    try:
        value_list.append(re.findall('(?<={}:\s)\w+'.format(parameter), f[0])[0])
    except:
        value_list.append('?')
for line in f:

    if 'DROP_KEEP' in line:
        value_list.append(float(re.findall('(?<=DROP_KEEP: )\d+.\d+', line)[0]))
        parameter_list.append('DROP_KEEP')

    if 'Model training took' in line:
        value_list.append(float(re.findall('(?<=Model training took, )\d+.\d+', line)[0]) / 60)
        parameter_list.append('RUNTIME')
    if 'test accuracy' in line:
        try:
            value_list.append(round(float(re.findall('(?<=test accuracy: )\d.\d+', line)[0]), 5))
        except:
            value_list.append('?')
        parameter_list.append('TEST_ACCURACY')
    if 'LEARNING' in line:
        try:
            value_list.append(re.findall('(?<=LEARNING: )\d.\d+', line)[0])
        except:
            value_list.append('?')
        parameter_list.append('LEARNING')

value_list.append(f[0].split(' - root -')[0])
parameter_list.append('START_TIME')

df = pd.DataFrame(data=value_list, index=parameter_list, columns=[''])

# save parameter as csv
df.to_csv(model_path + '/table.csv')

fig, ax1 = plt.subplots(figsize=(15, 10))

ax1.plot(step, steps_entro_val, 'bo', label='Val.')
ax1.plot(step, steps_entro_val, 'b', lw=2.5)
ax1.plot(step, steps_entro_train, 'yo', label='Train')
ax1.plot(step, steps_entro_train, 'y', lw=2.5)
ax1.set_ylim([0, 20])
ax1.set_yticks(list(range(0, 21, 4)))
ax1.set_xlabel('Steps').set_fontsize(20)
ax1.tick_params('x', labelsize=15)
ax1.legend(loc=6, fontsize=15)

# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Cross Entropy', color='b').set_fontsize(20)
ax1.tick_params('y', colors='b', labelsize=15)

ax2 = ax1.twinx()
ax2.set_ylim([0, 1])
ax2.plot(step, steps_ac_val, 'ro', label='Val.')
ax2.plot(step, steps_ac_val, 'r', lw=2.5)

ax2.plot(step, steps_ac_train, 'go', label='Train')
ax2.plot(step, steps_ac_train, 'g', lw=2.5)
ax2.set_ylabel('Accuracy', color='r').set_fontsize(20)

ax2.tick_params('y', colors='r', labelsize=15)

ax2.legend(loc=7, fontsize=15)
fig.tight_layout()

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(1.1, 0.75, 'HYPERPARAMETER:\n' + df.to_string(), transform=ax2.transAxes, fontsize=20, family='monospace',
         verticalalignment='top', bbox=props)
plt.savefig(model_path + '/graph_data', bbox_inches='tight', pad_inches=0.5)

# compress
shutil.make_archive(model_path,
                    'zip',
                    model_path)

# upload the compressed file to aws S3, aws cli is needed
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


