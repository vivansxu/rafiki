import os
import numpy as np
import PIL.Image
import glob

import TFRecordExporter
from rafiki.model import dataset_utils

def load(train_dataset_path, out_train_dataset_path):

    print('Loading User Dataset...')
    
    image_filenames = sorted(glob.glob(os.path.join(train_dataset_path, '*')))
    if len(image_filenames) == 0:
        print('Error: No input images found.')
        exit(1)
    
    img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1

    if img.shape[1] != resolution:
        print('Error: Input images must have the same width and height.')
        exit(1)
    if channels not in [1, 3]:
        print('Error: Input images must be stored as RGB or grayscale.')
        exit(1)
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        new_size = 2 ** int(np.floor(np.log2(resolution)))
        print('Resizing images to power-of-two resolution')
        train_dataset_path = _image_resize(train_dataset_path, new_size)
        image_filenames = sorted(glob.glob(os.path.join(train_dataset_path, '*')))
    
    with TFRecordExporter(out_train_dataset_path, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :]
            else:
                img = img.transpose(2, 0, 1)
            tfr.add_image(img)

    print('Train dataset TFRecord file is saved at {}'.format(out_train_dataset_path))

def _image_resize(train_dataset_path, new_size):
    image_list = os.listdir(train_dataset_path)
    new_path = os.path.join(train_dataset_path, 'resolution_power-of-two_version')
    for image_name in image_list:
        image = PIL.Image.open(os.path.join(train_dataset_path, image_name))
        image = image.resize((new_size, new_size), PIL.Image.ANTIALIAS)
        image.save(os.path.join(new_path, image_name), image.mode)
    
    assert np.asarray(Image.open(os.path.join(new_path, image_list[0]))).shape[0] == new_size
    return new_path

if __name__ == '__main__':
    load(
        train_dataset_path='',
        out_train_dataset_path='data/mnist_for_image_generation.zip'
    )


    
    