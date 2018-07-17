import boto3
import shutil
import os
import tables
import numpy as np
from random import shuffle
import glob

s3 = boto3.resource('s3')
bucket = s3.Bucket('imagesforcnn')


def zip_to_s3(from_dir, to_dir, extension, archive_name):

    if extension is 'zip':
        shutil.make_archive(os.path.join(os.path.join(to_dir, archive_name)), extension, from_dir)

    # Upload a new file
    data = open(os.path.join(to_dir, archive_name + '.{}'.format(extension)), 'rb')
    print('uploading {}.{} ... \r'.format(archive_name, extension))
    bucket.put_object(Key=archive_name + '.{}'.format(extension), Body=data)
    print('uploading {}.{} finished'.format(archive_name, extension))


def s3_to_data_set(hdf5_path,train_path, shuffle_data=True ):

    # read addresses and labels from the 'train' folder
    addrs = glob.glob(train_path)
    labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
    # to shuffle data
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)

    # Divide the data into 80% train, 20% test
    train_addrs = addrs[0:int(0.8 * len(addrs))]
    train_labels = labels[0:int(0.8 * len(labels))]
    test_addrs = addrs[int(0.8 * len(addrs)):]
    test_labels = labels[int(0.8 * len(labels)):]

    img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
    data_shape = (0, 224, 224, 3)
    # open a hdf5 file and create earrays
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
    test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)
    mean_storage = hdf5_file.create_earray(hdf5_file.root, 'train_mean', img_dtype, shape=data_shape)
    # create the label arrays and copy the labels data in them
    hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
    hdf5_file.create_array(hdf5_file.root, 'test_labels', test_labels)


    print()


def data_set_to_s3():
    print()


def load_data_set():
    hdf5_path = 'Cat vs Dog/dataset.hdf5'
    subtract_mean = False
    # open the hdf5 file
    hdf5_file = tables.open_file(hdf5_path, mode='r')
    # subtract the training mean
    if subtract_mean:
        mm = hdf5_file.root.train_mean[0]
        mm = mm[np.newaxis, ...]
    # Total number of samples
    data_num = hdf5_file.root.train_img.shape[0]
    print()
