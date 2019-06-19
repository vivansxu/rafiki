import pickle
import os
import numpy as np
import tensorflow as tf
import glob
import sys
import argparse
import threading
import six.moves.queue as Queue
import traceback
from PIL import Image
import inspect
import importlib
import imp
from collections import OrderedDict
import re
import datetime
import time
import bisect
import scipy.ndimage
import scipy.misc

from rafiki.model import BaseModel, InvalidModelParamsException, test_model_class, \
                        IntegerKnob, CategoricalKnob, dataset_utils, logger
from rafiki.constants import TaskType, ModelDependency

#----------------------------------------------------------------------------
# Implements Progressive Growing of GANs for image generation

class PG_GANs():

    @staticmethod
    def get_knob_config():
        return {
            'minibatch_base': FixedKnob(4)

            'lod_initial_resolution': FixedKnob(4)
            'D_repeats': IntegerKnob(1, 3)
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        np.random.seed(1000)
        print('Initializing TensorFlow...')

        tf_config = {}
        tf_config['graph_options.place_pruned_graph'] = True
        tf_config['gpu_options.allow_growth'] = False

        #TODO: any environment var needed?

        if tf.get_default_session() is None:
            tf.set_random_seed(np.random.randint(1 << 31))
            self._session = _create_session(config_dict=tf_config, force_as_default=True)

       
    def train(self, dataset_uri):
        training_set = _load_dataset(data_dir=dataset_uri, {'tfrecord_dir': './'})
        with tf.device('/gpu:0'):
            print('Constructing networks...')
            # TODO: implement Network Class
            # TODO: finish config_G, config_D
            G = Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config_G)


    def evaluate(self, dataset_uri):
        dataset = dataset_utils.load_dataset_of_image_files(dataset_uri)
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset])
        X = self._prepare_X(images)
        y = classes
        preds = self._clf.predict(X)
        accuracy = sum(y == preds) / len(y)
        return accuracy

    def predict(self, queries):
        X = self._prepare_X(queries)
        probs = self._clf.predict_proba(X)
        return probs.tolist()

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}

        # Save model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64
        
        return params

    def load_parameters(self, params):
        # Load model parameters
        clf_base64 = params.get('clf_base64', None)
        if clf_base64 is None:
            raise InvalidModelParamsException()
        
        clf_bytes = base64.b64decode(params['clf_base64'].encode('utf-8'))
        self._clf = pickle.loads(clf_bytes)

    # Creating TensorFlow Session
    def _create_session(config_dict=dict(), force_as_default=False):
        config = tf.ConfigProto()

        for key, value in config_dict.items():
            fields = key.split('.')
            obj = config
            for field in fields[:-1]:
                obj = getattr(obj, field)
            setattr(obj, fields[-1], value)

        session = tf.Session(config=config)

        if force_as_default:
            session._default_session = session.as_default()
            session._default_session.enforce_nesting = False
            session._default_session.__enter__()

        return session

    def _load_dataset(data_dir=None, **kwargs):
        adjusted_kwargs = dict(kwargs)
        if 'tfrecord_dir' in adjusted_kwargs and data_dir is not None:
            adjusted_kwargs['tfrecord_dir'] = os.path.join(data_dir, adjusted_kwargs['tfrecord_dir'])
        dataset = TFRecordDataset(**adjusted_kwargs)
        return dataset

    # Main Training Process
    def _train_progressive_gan(
        G_smoothing             = 0.99,
        D_repeats               = 1,
        minibatch_repeats       = 4,
        reset_opt_for_new_lod   = True,
        total_kimg              = 15000,
        mirror_augment          = False,
        drange_net              = [-1, 1]):

# Dataset Class for tfrecords files
class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,                   # dataset directory
        resolution          = None,     # None: autodetect
        label_file          = None,     # None: autodetect
        max_label_size      = 0,        # 0: no labels; 'full': full labels; n: n first label components
        repeat              = True,
        shuffle_mb          = 4096,     # 0: disable shuffling
        prefetch_mb         = 2048,     # 0: disable prefetching
        buffer_mb           = 256,
        num_threads         = 2):

        self.tfrecord_dir = tfrecord_dir
        self.resolution = None
        self.resolution_log2 = None
        self.shape = []                     # [c, h, w]
        self.dtype = 'uint8'
        self.dynamic_range = [0, 255]
        self.label_file = label_file
        self.label_size = None              # [component]
        self.label_dtype = None
        self._np_labels = None
        self._tf_minibatch_in = None
        self._tf_labels_var = None
        self._tf_labels_dataset = None
        self._tf_datasets = dict()
        self._tf_iterator = None
        self._tf_init_ops = dict()
        self._tf_minibatch_np = None
        self._cur_minibatch = -1
        self._cur_lod = -1

        assert os.path.isdir(self.tfrecord_dir)

        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) >= 1

        tfr_shapes = []
        for tfr_file in tfr_files:
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for record in tf.python_io.tf_record_iterator(tfr_file, tfr_opt):
                tfr_shapes.append(_parse_tfrecord_np(record).shape)
                break

        if self.label_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.labels')))
            if len(guess):
                self.label_file = guess[0]
        elif not os.path.isfile(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if os.path.isfile(guess):
                self.label_file = guess

        max_shape = max(tfr_shapes, key=lambda shape: np.prod(shape))
        self.resolution = resolution if resolution is not None else max_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]

        assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))
        assert all(lod in tfr_lods for lod in range(self.resolution_log2 - 1))

        assert max_label_size == 'full' or max_label_size >= 0
        self._np_labels = np.zeros([1<<20, 0], dtype=np.float32)
        if self.label_file is not None and max_label_size != 0:
            self._np_labels = np.load(self.label_file)
            assert self._np_labels.ndim == 2
        if max_label_size != 'full' and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name

        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            self._tf_minibatch_in = tf.placeholder(tf.int64, name='minibatch_in', shape=[])
            tf_labels_init = tf.zeros(self._np_labels.shape, self._np_labels.dtype)
            self._tf_labels_var = tf.Variable(tf_labels_init, name='labels_var')
            _set_vars({self._tf_labels_var: self._np_labels})
            self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)
            for tfr_file, tfr_shape, tfr_lod in zip(tfr_files, tfr_shapes, tfr_lods):
                if tfr_lod < 0:
                    continue
                dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
                dset = dset.map(_parse_tfrecord_tf, num_parallel_calls=num_threads)
                dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
                bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
                if shuffle_mb > 0:
                    dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
                if repeat:
                    dset = dset.repeat()
                if prefetch_mb > 0:
                    dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
                dset = dset.batch(self._tf_minibatch_in)
                self._tf_datasets[tfr_lod] = dset
            self.tf_record_iterator = tf.data.Iterator.from_structure(self._tf_datasets[0].output_types, self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) for lod, dset in self._tf_datasets.items()}

    # use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf()
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # get next minibatch as TensorFlow expressions
    def get_minibatch_tf(self):
        return self._tf_iterator.get_next()

    # get next minibatch as NumPy arrays
    def get_minibatch_np(self, minibatch_size, lod=0):
        self.configure(minibatch_size, lod)
        if self._tf_minibatch_np is None:
            self._tf_minibatch_np = self.get_minibatch_tf()
        return tf.get_default_session().run(self._tf_minibatch_np)

    # get random labels as TensorFlow expression
    def get_random_labels_tf(self, minibatch_size):
        if self.label_size > 0:
            return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 0, self._np_labels.shape[0], dtype=tf.int32))
        else:
            return tf.zeros([minibatch_size, 0], self.label_dtype)

    # get random labels as Numpy array
    def get_random_labels_np(self, minibatch_size):
        if self.label_size > 0:
            return self._np_labels[np.random.randint(self._np_labels.shape[0], size=[minibatch_size])]
        else:
            return np.zeros([minibatch_size, 0], self.label_dtype)

    # parse individual image from a tfrecords file as TensorFlow expression
    def _parse_tfrecord_np(record):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value
        data = ex.features.feature['data'].bytes_list.value[0]
        return np.fromstring(data, np.uint8).reshape(shape)

    # parse individual image from a tfrecords file as NumPy array
    def _parse_tfrecord_tf(record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)})
        data = tf.decode_raw(features['data'], tf.uint8)
        return tf.reshape(data, features['shape'])

class Network:
    def __init__(self,
        name=None,
        func=None,
        **static_kwargs):

        self._init_fields()     # TODO: implement _init_fields()
        self.name = name
        self.static_kwargs = dict(static_kwargs)

        module, self._build_func_name =     # TODO: implement networks.G_paper, D_paper

def _set_vars(var_to_value_dict):
    ops = []
    feed_dict = {}
    for var, value in var_to_value_dict.items():
        assert isinstance(var, tf.Tensor) or isinstance(var, tf.Variable) or isinstance(var, tf.Operation)
        try:
            setter = tf.get_default_graph().get_tensor_by_name(var.name.replace(':0', '/setter:0'))
        except KeyError:
            with tf.name_scope(var.name.split(':')[0] + '/'):
                with tf.control_dependencies(None):
                    setter = tf.assign(var, tf.placeholder(var.dtype, var.shape, 'new_value'), name='setter')
        ops.append(setter)
        feed_dict[setter.op.inputs[1]] = value
    tf.get_default_session().run(ops, feed_dict)

if __name__ == '__main__':
    '''test_model_class(
        model_file_path=__file__,
        model_class='SkDt',
        task=TaskType.IMAGE_CLASSIFICATION,
        dependencies={
            ModelDependency.SCIKIT_LEARN: '0.20.0'
        },
        train_dataset_uri='data/fashion_mnist_for_image_classification_train.zip',
        test_dataset_uri='data/fashion_mnist_for_image_classification_test.zip',
        queries=[
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 7, 0, 37, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 27, 84, 11, 0, 0, 0, 0, 0, 0, 119, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 88, 143, 110, 0, 0, 0, 0, 22, 93, 106, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 53, 129, 120, 147, 175, 157, 166, 135, 154, 168, 140, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 11, 137, 130, 128, 160, 176, 159, 167, 178, 149, 151, 144, 0, 0], 
            [0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 3, 0, 0, 115, 114, 106, 137, 168, 153, 156, 165, 167, 143, 157, 158, 11, 0], 
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 89, 139, 90, 94, 153, 149, 131, 151, 169, 172, 143, 159, 169, 48, 0], 
            [0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0, 98, 136, 110, 109, 110, 162, 135, 144, 149, 159, 167, 144, 158, 169, 119, 0], 
            [0, 0, 2, 2, 1, 2, 0, 0, 0, 0, 26, 108, 117, 99, 111, 117, 136, 156, 134, 154, 154, 156, 160, 141, 147, 156, 178, 0], 
            [3, 0, 0, 0, 0, 0, 0, 21, 53, 92, 117, 111, 103, 115, 129, 134, 143, 154, 165, 170, 154, 151, 154, 143, 138, 150, 165, 43], 
            [0, 0, 23, 54, 65, 76, 85, 118, 128, 123, 111, 113, 118, 127, 125, 139, 133, 136, 160, 140, 155, 161, 144, 155, 172, 161, 189, 62], 
            [0, 68, 94, 90, 111, 114, 111, 114, 115, 127, 135, 136, 143, 126, 127, 151, 154, 143, 148, 125, 162, 162, 144, 138, 153, 162, 196, 58], 
            [70, 169, 129, 104, 98, 100, 94, 97, 98, 102, 108, 106, 119, 120, 129, 149, 156, 167, 190, 190, 196, 198, 198, 187, 197, 189, 184, 36], 
            [16, 126, 171, 188, 188, 184, 171, 153, 135, 120, 126, 127, 146, 185, 195, 209, 208, 255, 209, 177, 245, 252, 251, 251, 247, 220, 206, 49], 
            [0, 0, 0, 12, 67, 106, 164, 185, 199, 210, 211, 210, 208, 190, 150, 82, 8, 0, 0, 0, 178, 208, 188, 175, 162, 158, 151, 11], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ]
        
    )'''
    a = PG_GANs()
    a.train()
