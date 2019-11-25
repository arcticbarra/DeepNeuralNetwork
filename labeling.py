from random import shuffle
import numpy as np
import glob
import h5py
import cv2

shuffle_data = True  # shuffle the addresses before saving
# address to where you want to save the hdf5 file
hdf5_path = 'datasets/train_hotdogvsnothotdog.hdf5'
hot_dog_train_path = 'hd_dataset/train/*.jpg'
# read addresses and labels from the 'train' folder
addrs = glob.glob(hot_dog_train_path)
labels = [0 if 'hotdog' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 60% train, 20% validation, and 20% train
train_addrs = addrs
train_labels = labels

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
# check the order of data and chose proper data shape to save images
if data_order == 'th':
    train_shape = (len(train_addrs), 3, 75, 75)
elif data_order == 'tf':
    train_shape = (len(train_addrs), 75, 75, 3)

# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels

mean = np.zeros(train_shape[1:], np.float32)
# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 750 images
    if i % 750 == 0 and i > 1:
        print('train data: {}/{}'.format(i, len(train_addrs)))
    # read an image and resize to (75, 75)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (75, 75), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))

# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()
