import os
import numpy as np
import gzip
import PIL.Image

import TFRecordExporter
from rafiki.model import dataset_utils

def load(train_images_url, train_labels_url, out_train_dataset_path):
    

    print('Downloading files...')
    train_images_file_path = dataset_utils.download_dataset_from_uri(train_images_url)
    train_labels_file_path = dataset_utils.download_dataset_from_uri(train_labels_url)


    print('Loading MNIST...')
    with gzip.open(train_images_file_path, 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    with gzip.open(train_labels_file_path, 'rb') as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
    images = images.reshape(-1, 1, 28, 28)
    images = np.pad(images, [(0,0), (0,0), (2,2), (2,2)], 'constant', constant_value=0)

    assert images.shape == (60000, 1, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(out_train_dataset_path, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size()):
            tfr.add_image(image[order[idx]])
        tfr.add_labels(onehot[order])

    print('Train dataset TFRecord file is saved at {}'.format(out_train_dataset_path))

if __name__ == '__main__':
    load(
        train_images_url='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        train_labels_url='http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        out_train_dataset_path='data/mnist_for_image_generation'
    )


    
    