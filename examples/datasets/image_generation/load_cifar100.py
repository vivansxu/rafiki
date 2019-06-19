import os
import numpy as np
import pickle
import PIL.Image
import tarfile

import TFRecordExporter
from rafiki.model import dataset_utils

def load(train_dataset_url, out_train_dataset_path):
    
    print('Downloading files...')
    train_dataset_file_path = dataset_utils.download_dataset_from_uri(train_dataset_url)

    print('Extracting .tar file...')
    tar_file = tarfile.open(train_dataset_file_path)
    tar_file.extractall()

    train_dataset_file_path = os.path.join(os.path.dirname(train_dataset_file_path), 'cifar-10-python')

    print('Loading CIFAR-100...')
    images = []
    labels = []

    with open(os.path.join(train_dataset_file_path, 'train'), 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    images = data['data'].reshape(-1, 3, 32, 32)
    labels = np.array(data['fine_labels'])

    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype == np.int32
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 99

    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(out_train_dataset_path, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])
        
    print('Train dataset TFRecord file is saved at {}'.format(out_train_dataset_path))

if __name__ == '__main__':
    load(
        train_dataset_url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
        out_train_dataset_path='data/cifar100_for_image_generation'
    )


    
    