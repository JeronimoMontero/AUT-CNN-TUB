# coding: utf-8

#########################################
#
#  source ~/envs/tensorflow/bin/activate
#
#########################################

import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import sys

assert (len(sys.argv) > 1), 'Image path is missing'
assert os.path.exists(sys.argv[1]), 'Image path does not exist'

# get path to image
file_path = sys.argv[1]

home = os.getcwd().split('AUT-CNN-TUB')[0]

# path to model
path = os.path.join(home, 'AUT-CNN-TUB/Data/Models/model_merge_3CONV_3P_MEMORY_56_b50_s53256_2018-09-26_18-07')
export_dir = os.path.join(path, 'simple_save')


# load model
predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

# get pixel-size from model
new_size = int(predict_fn.feed_tensors['x'].shape[2])


# load images and center/scale it
def get_part_from_image(image, scale=1.1):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = imgray.mean()
    form_factor = imgray.shape[1] / 1280

    blurred = cv2.bilateralFilter(imgray, 10, brightness / form_factor, brightness / form_factor)
    thresh = cv2.Canny(blurred, brightness / 4, brightness / 4)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.dilate(thresh, kernel, iterations=5)

    _, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)

    scaled_rad = int(radius * scale)
    try:
        img_cutted = image[center[1] - scaled_rad: center[1] + scaled_rad,
                     center[0] - scaled_rad: center[0] + scaled_rad, :]
    except:
        img_cutted = image[center[1] - radius: center[1] + radius,
                     center[0] - radius: center[0] + radius, :]

    return img_cutted


img_name = file_path.split('/')[-1].replace('.jpg', '')
image = cv2.imread(file_path)

img_centerd = get_part_from_image(image, scale=1.5)

if img_centerd is not None:
    try:
        img_resized = cv2.resize(img_centerd, (new_size, new_size))
    except:
        print('Image {}.jpg is to small.'.format(img_name))
        img_resized = None

    assert img_resized is not None
    assert img_resized.var() > 200, 'Image {}.jpg does not have the necessary variance.'.format(img_name)


img_cutted = cv2.resize(img_resized, (new_size, new_size))

# gray
if predict_fn.graph.get_tensor_by_name('Variable:0').shape[2] == 1:

    img_cutted = cv2.cvtColor(img_cutted, cv2.COLOR_BGR2GRAY)
    img_cutted = img_cutted.reshape(img_cutted.shape[0], img_cutted.shape[1], 1)

    predictions = predict_fn({'x': np.array([img_cutted]).astype(dtype='float64'),
                              'keep_prob': 1})

# color
else:
    predictions = predict_fn({'x': np.array(img_cutted).reshape(1, new_size, new_size, 3).astype(dtype='float64'),
                              'keep_prob': 1})

# translator
label_list = ['1', '2', '3', '4', '5', '6', '7.1', '7.2', '8', '9', '10', '11', '12', '13', '14', '15']

with tf.Session() as sess:
    confidence = sess.run(tf.nn.softmax(logits=predictions['full'].astype(dtype='float64')))

print('\nprobably part:', label_list[int(predictions['predict'][0])], '\n')


for n, confi in enumerate((np.round_(confidence, 3) * 100).astype(int)[0].tolist()):
    print('{}\t{}%'.format(label_list[n], confi))

print('\n')
im = Image.fromarray(cv2.resize(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), (500, 500)))

im.show(title=img_name)
