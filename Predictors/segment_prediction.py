import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf
import sys

assert (len(sys.argv) > 1), 'Image path is missing'
assert os.path.exists(sys.argv[1]), 'Image path does not exist'

file_path = sys.argv[1]

img_name = file_path.split('/')[-1]

home = os.getcwd().split('AUT-CNN-TUB')[0]


# model path
path = os.path.join(home, 'AUT-CNN-TUB/Data/Models/model_merge_3CONV_3P_MEMORY_56_b50_s53256_2018-09-26_18-07')
export_dir = os.path.join(path, 'simple_save')

home = os.getcwd().split('AUT-CNN-TUB')[0]

image = cv2.imread(file_path)

kernel_faktor = int(image.shape[1] * 0.018 - 8)

blurred = cv2.bilateralFilter(image, 10, 50, 50)
imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

form_factor = imgray.shape[0]/1024

brightness1 = imgray.mean()/kernel_faktor*4

thresh = cv2.Canny(imgray, brightness1, brightness1)

kernel = np.ones((int(5 * form_factor), int(5 * form_factor)), np.uint8)
dilation = cv2.dilate(thresh, kernel, iterations=2)


h, b = thresh.shape
faktor = 0.5
_, contours, _ = cv2.findContours(dilation,
                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(imgray.shape,np.uint8)
sub_contour = []
for c in contours:
    if ((c[:, :, 0].max()-c[:, :, 0].min()) > b * faktor and (
            c[:, :, 1].max()-c[:, :, 1].min()) > h * faktor):
        sub_contour.append(c)

area = (cv2.drawContours(mask, sub_contour, 0, (1, 1, 1), -1))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (int(20 * form_factor),
                                     int(20 * form_factor)))
erosion = cv2.erode(area, kernel, iterations=2)

area_new = erosion

_, contours, _ = cv2.findContours(area_new,
                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    if ((c[:, :, 0].max()-c[:, :, 0].min()) > b * faktor and (
            c[:, :, 1].max()-c[:, :, 1].min()) > h * faktor):
        sub_contour.append(c)
frame = sub_contour

if area.sum() < 50000 / form_factor:
    print(img_name + ' bad image, no frame detected')

else:
    blurred = cv2.bilateralFilter(imgray, 10, 10, 10)

    brightness = (imgray * area_new).mean() / 2  * form_factor
    thresh2 = cv2.Canny(blurred, 30, 30)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilation1 = cv2.dilate(thresh2*area_new, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(dilation1, kernel, iterations=12)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    dilation2 = cv2.dilate(erosion, kernel, iterations=1)

    sub_contour = []
    _, contours, _ = cv2.findContours(dilation2, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    sub_contour = contours

    center = []
    counter = 0
    for c in sub_contour:
        counter += 1
        # compute the center of the contour
        M = cv2.moments(c)
        try:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center.append([cx, cy])
        except:
            print('moment is zero')

    center = np.array(center)

    to_drop = []
    for e, c in enumerate(sub_contour):
        for n, test in enumerate(sub_contour):
            if n != e:
                if c[:, :, 0].max() < test[:, :, 0].max():
                    if c[:, :, 0].min() > test[:, :, 0].min():
                        if c[:, :, 1].max() < test[:, :, 1].max():
                            if c[:, :, 1].min() > test[:, :, 1].min():
                                if len(c) is not len(test):
                                    to_drop.append(e)

    sub_con = np.delete(sub_contour, to_drop).tolist()

    center = np.delete(center, to_drop, 0)
    if len(center) is not 17:
        print('ERROR: ', len(center), 'parts detectet', img_name)

    img = cv2.drawContours(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), sub_con, -3, (255, 153, 51), int(3))

    for e, ce in enumerate(center, 1):
        cv2.putText(img, str(e), (ce[0], ce[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (57, 255, 20), int(2))

    cv2.drawContours(img, frame, 0, (57, 255, 20), int(3))

    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

    circles = image
    centers = []
    radien = []
    for c in sub_con:
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        centers.append(center)
        radius = int(radius)
        radien.append(radius)
        circles = cv2.circle(circles, center, radius, (0, 255, 0), 2)
        circles.size

    # if you want see the the circles uncomment
    # im = Image.fromarray(cv2.resize(circles, (500, int(500 * circles.shape[0]/circles.shape[1]))))
    # im.show()

    label_list = ['1', '2', '3', '4', '5', '6', '7.1', '7.2', '8', '9', '10', '11', '12', '13', '14', '15']
    predict_fn = tf.contrib.predictor.from_saved_model(export_dir)
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

    label_lit = []
    center_list = []

    for e, _ in enumerate(sub_con):

        radius = radien[e]
        center = centers[e]

        scale = 1.5
        scaled_rad = radius * scale

        if scaled_rad < 14:
            scaled_rad = 14

        try:
            img_cutted = image[
                         center[1] - scaled_rad: center[1] + scaled_rad,
                         center[0] - scaled_rad: center[0] + scaled_rad,
                         :]
        except:
            img_cutted = image[
                         center[1] - radius: center[1] + radius,
                         center[0] - radius: center[0] + radius,
                         :]

            new_size = int(predict_fn.feed_tensors['x'].shape[2])
            img_cutted = cv2.resize(img_cutted, (new_size, new_size))

            if predict_fn.graph.get_tensor_by_name('Variable:0').shape[2] == 1:

                img_cutted = cv2.cvtColor(img_cutted, cv2.COLOR_BGR2GRAY)
                img_cutted = img_cutted.reshape(img_cutted.shape[0], img_cutted.shape[1], 1)

                predictions = predict_fn({'x': np.array([img_cutted]), 'keep_prob': 1})
                label_lit.append(label_list[int(predictions['predict'][0])])
            else:
                predictions = predict_fn({'x': np.array(img_cutted).reshape(new_size, new_size, 3), 'keep_prob': 1})
                label_lit.append(label_list[int(predictions['predict'][0])])

            center_list.append(center)
        for n, center in enumerate(center_list):
            cv2.putText(image, label_lit[n], (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (57, 255, 20), 7)

       # if you want see the segments uncoment
       # im = Image.fromarray(cv2.resize(img_cutted, (500, int(500 * circles.shape[0] / circles.shape[1]))))
       # im.show()

    im = Image.fromarray(cv2.resize(image, (500, int(500 * circles.shape[0]/circles.shape[1]))))
    im.show()

