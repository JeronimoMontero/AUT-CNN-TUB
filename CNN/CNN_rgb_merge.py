# coding: utf-8


############################################
#                                          #
#   source ~/envs/tensorflow/bin/activate   #
#                                          #
############################################


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
import botocore

from layers import conv_layer, max_pool_2x2, full_layer

# Create a resource service client, and select bucket
s3 = boto3.resource('s3')
bucket = s3.Bucket('imagesforcnn')

start = time.time()

home = os.getcwd().split('AUT-CNN-TUB')[0]
DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

MINIBATCH_SIZE  = 100
STEPS = 5000
PIXEL = 28
COLOR = 3 # 3 or 1
CFAK = 6
DROP = 0.8

CONV = int(round(PIXEL/CFAK, 0))

model_path = os.path.join(home,'AUT-CNN-TUB', 'Data', 'Models',
                          'model_merge_{}_b{}_s{}_{}'.format(PIXEL,
		                                        MINIBATCH_SIZE, STEPS, DATE)) 
if os.path.exists(model_path) is False:                   
    os.makedirs(model_path)

# create logger
logger = logging.getLogger()
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

logger.info('MINIBATCH_SIZE: {}, STEPS: {}, PIXEL: {}, COLOR: {}, CFAK: {}, DROP: {}'.format(MINIBATCH_SIZE, STEPS, PIXEL, COLOR, CFAK, DROP))

if COLOR is 3:
    Gray = False
elif COLOR is 1:
    Gray = True
else:
    logger.error('COLOR has to be 1 or 3')

assert COLOR in [3,1]

test_path  = os.path.join(home,'AUT-CNN-TUB/Data/aws_test/TF_Images_merge_{}/test/'.format(PIXEL))
train_path = os.path.join(home,'AUT-CNN-TUB/Data/aws_test/TF_Images_merge_{}/train/'.format(PIXEL))

test_number = len(os.listdir(test_path))
train_number = len(os.listdir(train_path))

labels = ['01.0', '02.0', '03.0', '04.0', '05.0', '06.0', '07.1', '07.2',
         '08.0', '09.0','10.0','11.0', '12.0','13.0','14.0', '15.0']
position_dict = {k: v for v, k in enumerate(labels)}
label_dict = {v : k  for v, k in enumerate(labels)}


def label_to_binary(position_dict, label):
    z = np.zeros(len(position_dict), dtype=int)
    z[position_dict[label]] = 1
    return z

class DataGetter:
    def __init__(self, path, GRBtoGray=False, flatten=False): 
        self.path  = path
        self.GRBtoGray  = GRBtoGray
        self.flatten  = flatten
        self.batchindices = None
        
        self.data_path = os.path.join(self.path, '*g') 
        self.files = glob.glob(self.data_path)
        self.num_imag = len(self.files)
        self.Data = list(range(0, self.num_imag))

    def get_batch(self,size):
        
        if len(self.Data) > size:
            self.batchindices = list(np.random.choice(self.Data, size=size, replace=False))
            for x in self.batchindices:
                self.Data.remove(x) 
                
        elif len(self.Data) == size: # new epoche
            self.batchindices = np.random.choice(self.Data, size=size, replace=False)  
            self.Data = list(range(0, self.num_imag))

            
        else: # new epoche
            self.batchindices = np.random.choice(self.Data, size=len(self.Data), replace=False)  
            self.Data = list(range(0, self.num_imag))



            

        label_list =[]
        img_list = []
        file_name_list = []
        for i, file_path in enumerate(self.files):
            if i in self.batchindices:
                label_str = file_path.split('/')[-1][:4]
                label_bin = label_to_binary(position_dict, label_str)

                if self.GRBtoGray == True:
                    img = cv2.imread(file_path, flags=0)
                else:
                    img = cv2.imread(file_path, flags=1)

                if self.flatten == True:
                    img_list.append(img.flatten())
                    label_list.append(label_bin)
                    file_name_list.append(file_path.split('/')[-1])
                else:
                    img_list.append(img)
                    label_list.append(label_bin)
                    file_name_list.append(file_path.split('/')[-1])

        img_list = np.array(img_list)
        label_list = np.array(label_list)


        return img_list, label_list, file_name_list
    
test_img = DataGetter(test_path, Gray, False)
train_img = DataGetter(train_path, Gray, False)



x = tf.placeholder(tf.float32, shape=[None, PIXEL, PIXEL, COLOR])
y_ = tf.placeholder(tf.float32, shape=[None, len(position_dict)])


conv1 = conv_layer(x, shape=[CONV, CONV, COLOR, 64])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[CONV, CONV, 64, 128])
conv2_pool = max_pool_2x2(conv2)

new_size = int(PIXEL/4*PIXEL/4*128)

if PIXEL%2 is not 0:
    logger.warning('potential issue with the pixel size')	

conv2_flat = tf.reshape(conv2_pool, [-1, new_size])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, len(position_dict))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epoch_counter = 0 

with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            X_batch, y_batch, file_name = test_img.get_batch(MINIBATCH_SIZE)

            if i % 50 == 0:

                train_accuracy = sess.run(accuracy, feed_dict={x: X_batch, y_: y_batch,
                                                               keep_prob: 1.0})
                cross_entro = sess.run(cross_entropy, feed_dict={x: X_batch, y_: y_batch,
                                                             keep_prob: 1.0})

                logger.info("step {}, training accuracy {}".format(i, train_accuracy))
                logger.info("step {}, cross entropy {}".format(i, cross_entro))
            
   # epoch
            if round((i*MINIBATCH_SIZE)/train_number, 0) > epoch_counter:

                epoch_counter += 1
                
                test_sum = 0
                cross_entro_sum = 0
                for test in range(0,test_number, 100):

                    tX_batch, ty_batch, file_name = test_img.get_batch(100) # 

                    test_accuracy = sess.run(accuracy, feed_dict={x: tX_batch, y_: ty_batch, keep_prob: 1.0}) 
                    cross_entro = sess.run(cross_entropy, feed_dict={x: tX_batch, y_: ty_batch, keep_prob: 1.0})


                    test_sum = (test_sum + test_accuracy) /2
                    cross_entro_sum = (cross_entro_sum + cross_entro)/2

                logger.info("epoch {}, test accuracy {}".format(epoch_counter, test_accuracy))
                logger.info("epoch {}, training accuracy {}".format(epoch_counter, train_accuracy))
                logger.info("epoch {}, cross entropy {}".format(epoch_counter, cross_entro))

            sess.run(train_step, feed_dict={x: X_batch, y_: y_batch, keep_prob: DROP})
        test_sum = 0
        for test in range(0,test_number, 100):

            tX_batch, ty_batch, file_name = test_img.get_batch(100) # 

            test_accuracy = sess.run(accuracy, feed_dict={x: tX_batch, y_: ty_batch, keep_prob: 1.0}) 

            test_sum = (test_sum + test_accuracy) / 2

        logger.info("test accuracy: {}".format(test_sum))

        model_path_file  = os.path.join(model_path, 'model_merge_{}_b{}_s{}_{}.ckpt'.format(
		                                PIXEL, MINIBATCH_SIZE, STEPS, DATE))
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path_file)
        logger.info("Model saved in path: %s" % save_path)

end = time.time()

logger.info('Model training tock, {} secondes'.format(str(end-start)))

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
logger.info('Model training tock, {} secondes'.format(str(end-start)))
