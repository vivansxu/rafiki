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
        num_gpus


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

        training_set = _load_dataset(data_dir=dataset_uri, {'tfrecord_dir': './'})
        with tf.device('/gpu:0'):
            print('Constructing networks...')
            # TODO: finish config_G, config_D
            G = Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config_G)
            D = Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config_D)
            Gs = G.clone('Gs')
            Gs_update_op = Gs.setup_as_moving_average_of(G, beta=G_smoothing)

        print('Building TensorFlow graph...')
        with tf.name_scope('Inputs'):
            lod_in = tf.placeholder(tf.float32, name='lod_in', shape=[])
            lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
            minibatch_in = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
            minibatch_split = minibatch_in / num_gpus       # TODO: enter num_gpus
            reals, labels = training_set.get_minibatch_tf()
            reals_split = tf.split(reals, num_gpus)
            labels_split = tf.split(labels, num_gpus)
        G_opt = Optimizer(name='TrainG', learning_rate=lrate_in, **config_G_opt)    # TODO: complete config_G_opt; Optimizer

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

        self._init_fields()     
        self.name = name
        self.static_kwargs = dict(static_kwargs)

        self._build_func_name = func
        self._build_func = getattr(Network, self._build_func_name)

        self._init_graph()
        self.reset_vars()

    def _init_fields(self):
        self.name               = None
        self.scope              = None
        self.static_kwargs      = dict()
        self.num_inputs         = 0
        self.num_outputs        = 0
        self.input_shapes       = [[]]
        self.output_shapes      = [[]]
        self.input_shape        = []
        self.output_shape       = []
        self.input_templates    = []
        self.output_templates   = []
        self.input_names        = []
        self.output_names       = []
        self.vars               = OrderedDict()
        self.trainables         = OrderedDict()
        self._build_func        = None
        self._build_func_name   = None
        self._run_cache         = dict()

    def _init_graph(self):
        self.input_names = []
        for param in inspect.signature(self._build_func).parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD and param.default is param.empty:
                self.input_names.append(param.name)
        self.num_inputs = len(self.input_names)
        assert self.num_inputs >= 1

        if self.name is None:
            self.name = self._build_func_name
        self.scope = tf.get_default_graph().unique_name(self.name.replace('/', '_'), mark_as_used=False)

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            assert tf.get_variable_scope().name == self.scope
            with tf.name_scope(self.scope + '/'):
                with tf.control_dependencies(None):
                    self.input_templates = [tf.placeholder(tf.float32, name=name) for name in self.input_names]
                    out_expr = self._build_func(*self.input_templates, is_template_graph=True, **self.static_kwargs)

        assert isinstance(out_expr, tf.Tensor) or isinstance(out_expr, tf.Variable) or isinstance(out_expr, tf.Operation) or isinstance(out_expr, tuple)
        self.output_templates = [out_expr] if (isinstance(out_expr, tf.Tensor) or isinstance(out_expr, tf.Variable) or isinstance(out_expr, tf.Operation)) else list(out_expr)
        self.output_names = [t.name.split('/')[-1].split(':')[0] for t in self.output_templates]
        self.num_outputs = len(self.output_templates)
        assert self.num_outputs >= 1

        self.input_shapes = [[dim.value for dim in t.shape] for t in self.input_templates]
        self.output_shapes = [[dim.value for dim in t.shape] for t in self.output_templates]
        self.input_shape = self.input_shapes[0]
        self.output_shape = self.output_shapes[0]
        self.vars = OrderedDict([(self.get_var_localname(var), var) for var in tf.global_variables(self.scope + '/')])
        self.trainables = OrderedDict([(self.get_var_localname(var), var) for var in tf.trainable_variables(self.scope + '/')])


    def reset_vars(self):
        run([var.initializer for var in self.vars.values()])

    def run(self, *in_arrays,
        return_as_list      = False,
        print_progress      = False,
        minibatch_size      = None,
        num_gpus            = 1,
        out_mul             = 1.0,
        out_add             = 0.0,
        out_shrink          = 1,
        out_dtype           = None,
        **dynamic_kwargs):

        assert len(in_arrays) == self.num_inputs
        num_items = in_arrays[0].shape[0]
        if minibatch_size is None:
            minibatch_size = num_items
        key = str([list(sorted(dynamic_kwargs.items())), num_gpus, out_mul, out_add, out_shrink, out_dtype])

        if key not in self._run_cache:
            with tf.name_scope((self.scope + '/Run') + '/'), tf.control_dependencies(None):
                in_split = list(zip(*[tf.split(x, num_gpus) for x in self.input_templates]))
                out_split = []

                for gpu in range(num_gpus):
                    with tf.device('/gpu:%d' % gpu):
                        out_expr = self.get_output_for(*in_split[gpu], return_as_list=True, **dynamic_kwargs)
                        if out_mul != 1.0:
                            out_expr = [x * out_mul for x in out_expr]
                        if out_add != 0.0:
                            out_expr = [x + out_add for x in out_expr]
                        if out_shrink > 1:
                            ksize = [1, 1, out_shrink, out_shrink]
                            out_expr = [tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') for x in out_expr]
                        if out_dtype is not None:
                            if tf.as_dtype(out_dtype).is_integer:
                                out_expr = [tf.round(x) for x in out_expr]
                            out_expr = [tf.saturate_cast(x, out_dtype) for x in out_expr]
                        out_split.append(out_expr)
                self._run_cache[key] = [tf.concat(outputs, axis=0) for outputs in zip(*out_split)]

        out_expr = self._run_cache[key]
        out_arrays = [np.empty([num_items] + [dim.value for dim in expr.shape][1:], expr.dtype.name) for expr in out_expr]
        for mb_begin in range(0, num_items, minibatch_size):
            if print_progress:
                print('\r%d / %d' % (mb_begin, num_items), end='')
            mb_end = min(mb_begin + minibatch_size, num_items)
            mb_in = [src[mb_begin : mb_end] for src in in_arrays]
            mb_out = tf.get_default_session().run(out_expr, dict(zip(self.input_templates, mb_in)))
            for dst, src in zip(out_arrays, mb_out):
                dst[mb_begin : mb_end] = src
        
        if print_progress:
            print('\r%d / %d' % (num_items, num_items))
        if not return_as_list:
            out_arrays = out_arrays[0] if len(out_arrays) == 1 else tuple(out_arrays)
        return out_arrays

    def setup_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        assert isinstance(src_net, Network)
        with tf.name_scope(self.scope + '/'):
            with tf.name_scope('MovingAvg'):
                ops = []
                for name, var in self.vars.items():
                    if name in src_net.vars:
                        cur_beta = beta if name in self.trainables else beta_nontrainable
                        new_value = _lerp(src_net.vars[name], var, cur_beta)
                        ops.append(var.assign(new_value))
                return tf.group(*ops)

    def clone(self, name=None):
        net = object.__new__(Network)
        net._init_fields()
        net.name = name if name is not None else self.name
        net.static_kwargs = dict(self.static_kwargs)
        net._build_func_name = self._build_func_name
        net._build_func = self._build_func
        net._init_graph()
        net.copy_vars_from(self)
        return net

    def get_var_localname(self, var_or_globalname):
        assert isinstance(var_or_globalname, tf.Tensor) or isinstance(var_or_globalname, tf.Variable) or isinstance(var_or_globalname, tf.Operation) or isinstance(var_or_globalname, str)
        globalname = var_or_globalname if isinstance(var_or_globalname, str) else var_or_globalname.name
        assert globalname.startswith(self.scope + '/')
        localname =globalname[len(self.scope) + 1:]
        localname = localname.split(':')[0]
        return localname
        
    def get_output_for(self, *in_expr, return_as_list=False, **dynamic_kwargs):
        assert len(in_expr) == self.num_inputs
        all_kwargs = dict(self.static_kwargs)
        all_kwargs.update(dynamic_kwargs)

        with tf.variable_scope(self.scope, reuse=True):
            assert tf.get_variable_scope().name == self.scope
            named_inputs = [tf.identify(expr, name=name) for expr, name in zip(in_expr, self.input_names)]
            out_expr = self._build_func(*named_inputs, **all_kwargs)
        assert isinstance(out_expr, tf.Tensor) or isinstance(out_expr, tf.Variable) or isinstance(out_expr, tf.Operation) or isinstance(out_expr, tuple)

        if return_as_list:
            out_expr = [out_expr] if (isinstance(out_expr, tf.Tensor) or isinstance(out_expr, tf.Variable) or isinstance(out_expr, tf.Operation)) else list(out_expr)
        return out_expr

    def copy_vars_from(self, src_net):
        assert isinstance(src_net, Network)
        name_to_value = run({name: src_net.find_var(name) for name in self.trainables.keys()})
        _set_vars({self.find_var(name): value for name, value in name_to_value.items()})

    def find_var(self, var_or_localname):
        assert isinstance(var_or_localname, tf.Tensor) or isinstance(var_or_localname, tf.Variable) or isinstance(var_or_localname, tf.Operation) or isinstance(var_or_localname, str)
        return self.vars[var_or_localname] if isinstance(var_or_localname, str) else var_or_localname


    # Generator network
    def G_paper(self,
        latents_in,                     # latent vectors [minibatch, latent_size]
        labels_in,                      # labels [minibatch, label_size]
        num_channels        = 1,
        resolution          = 32,
        label_size          = 0,        # 0: no labels
        fmap_base           = 8192,     # overall multiplier for num of feature maps
        fmap_decay          = 1.0,      # log2 feature map reduction when doubling resolution
        fmap_max            = 512,      # max num of feature map in any layer
        latent_size         = None,     # None: min(fmap_base, fmap_max)
        normalize_latents   = True,     # normalize latent vectors before feeding to network
        use_wscale          = True,     # enable equalized learning rate
        use_pixelnorm       = True,     # enable pixelwise feature vector normalization
        pixelnorm_epsilon   = 1e-8,
        use_leakyrelu       = True,     # False: ReLU
        dtype               = 'float32',
        fused_scale         = True,     # True: fused upscale2d+conv2d; False: separate upscale2d layers
        structure           = None,     # None: select automatically; 'linear': human-readable; 'recursive': efficient
        is_template_graph   = False,    # False: actual evaluation; True: template graph constructed by the class
        **kwargs):
        
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        def PN(x): return _pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x

        if latent_size is None: latent_size = nf(0)
        if structure is None: structure = 'linear' if is is_template_graph else 'recursive'
        act = _leaky_relu if use_leakyrelu else tf.nn.relu

        latents_in.set_shape([None, latent_size])       # TODO: set_shape() ?
        labels_in.set_shape([None, label_size])
        combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
        lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

        def block(x, res):  # res = 2..resolution_log2
            with tf.variable_scope('%dx%d' % (2**res, 2**res)):
                if res == 2:    # 4x4
                    if normalize_latents: x = _pixel_norm(x, epsilon=pixelnorm_epsilon)
                    with tf.variable_scope('Dense'):
                        x = _dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale)
                        x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                        x = PN(act(_apply_bias(x)))
                    with tf.variable_scope('Conv'):
                        x = PN(act(_apply_bias(_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                else:   # 8x8 and up
                    if fused_scale:
                        with tf.variable_scope('Conv0_up'):
                            x = PN(act(_apply_bias(_upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                    else:
                        x = _upscale2d(x)
                        with tf.variable_scope('Conv0'):
                            x = PN(act(_apply_bias(_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                    with tf.variable_scope('Conv1'):
                        x = PN(act(_apply_bias(_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                return x

        def torgb(x, res):
            lod = resolution_log2 - res
            with tf.variable_scope('ToRGB_lod%d' % lod):
                return _apply_bias(_conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

        if structure == 'linear':
            x = block(x, res)
            images_out = torgb(x, 2)
            for res in range(3, resolution_log2 + 1):
                lod = resolution_log2 - res
                x = block(x, res)
                img = torgb(x, res)
                images_out = _upscale2d(images_out)
                with tf.variable_scope('Grow_lod%d' % lod):
                    images_out = _lerp_clip(img, images_out, lod_in - lod)

        if structure == 'recursive':
            def grow(x, res, lod):
                y = block(x, res)
                img = lambda: _upscale2d(torgb(y, res), 2**lod)
                if res > 2:
                    img = _cset(img, (lod_in > lod), lambda: _upscale2d(_lerp(torgb(y, res), _upscale2d(torgb(x, res-1)), lod_in - lod), 2**lod))
                if lod > 0:
                    img = _cset(img, (lod_in < lod), lambda: grow(y, res+1, lod-1))
                return img()
            images_out = grow(combo_in, 2, resolution_log2 - 2)

        assert images_out.dtype == tf.as_dtype(dtype)
        images_out = tf.identify(images_out, name='image_out')
        return images_out

    # Discriminator network
    def D_paper(self,
        images_in,
        num_channels        = 1,
        resolution          = 32,
        label_size          = 0,
        fmap_base           = 8192,
        fmap_decay          = 1.0,
        fmap_max            = 512,
        use_wscale          = True,
        mbstd_group_size    = 4,        # 0: disable; group size for the minibatch standard deviation layer
        dtype               = 'float32',
        fused_scale         = True,
        structure           = None,
        is_template_graph   = False,
        **kwargs):

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        if structure is None: structure = 'linear' if is_template_graph else 'recursive'
        act = _leaky_relu

        images_in.set_shape([None, num_channels, resolution, resolution])
        images_in = tf.cast(images_in, dtype)
        lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

        def fromrgb(x, res):
            with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
                return act(_apply_bias(_conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
        
        def block(x, res):
            with tf.variable_scope('%dx%d' % (2**res, 2**res)):
                if res >= 3:
                    with tf.variable_scope('Conv0'):
                        x = act(_apply_bias(_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                    if fused_scale:
                        with tf.variable_scope('Conv1_down'):
                            x = act(_apply_bias(_conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    else:
                        with tf.variable_scope('Conv1'):
                            x = act(_apply_bias(_conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                        x = _downscale2d(x)
                else:
                    if mbstd_group_size > 1:
                        x = _minibatch_stddev_layer(x, mbstd_group_size)
                    with tf.variable_scope('Conv'):
                        x = act(_apply_bias(_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                    with tf.variable_scope('Dense0'):
                        x = act(_apply_bias(_dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                    with tf.variable_scope('Dense1'):
                        x = _apply_bias(_dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
                return x

        if structure == 'linear':
            img = images_in
            x = fromrgb(img, resolution_log2)
            for res in range(resolution_log2, 2, -1):
                lod = resolution_log2 - res
                x = block(x, res)
                img = _downscale2d(img)
                y = fromrgb(img, res - 1)
                with tf.variable_scope('Grow_lod%d' % lod):
                    x = _lerp_clip(x, y, lod_in - lod)
            combo_out = block(x, 2)

        if structure == 'recursive':
            def grow(res, lod):
                x = lambda: fromrgb(_downscale2d(images_in, 2**lod), res)
                if lod > 0:
                    x = _cset(x, (lod_in < lod), lambda: grow(res+1, lod-1))
                x = block(x(), res); y = lambda: X
                if res > 2:
                    y = _cset(y, (lod_in > lod), lambda: _lerp(x, fromrgb(_downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
                return y()
            combo_out = grow(2, resolution_log2 - 2)
        
        assert combo_out.dtype == tf.as_dtype(dtype)
        scores_out = tf.identify(combo_out[:, :1], name='scores_out')
        labels_out = tf.identify(combo_out[:, 1:], name='labels_out')
        return scores_out, labels_out

    # same as tf.nn.leaky_relu, but supports FP16
    def _leaky_relu(self, x, alpha=0.2):
        with tf.name_scope('LeakyRelu'):
            alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
            return tf.maximum(x * alpha, x)

    # pixelwise feature vector normalization
    def _pixel_norm(self, x, epsilon=1e-8):
        with tf.variable_scope('PixelNorm'):
            return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

    # fully connected layer
    def _dense(self, x, fmaps, gain=np.sqrt(2), use_wscale=False):
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
        w = _get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, x.dtype)
        return tf.matmul(x, w)

    # get/create weight tensor for a convolutional/fully-connected layer
    def _get_weight(self, shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
        if fan_in is None: fan_in = np.prod(shape[:-1])
        std = gain / np.sqrt(fan_in)
        if use_wscale:
            wscale = tf.constant(np.float32(std), name='wscale')
            return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
        else:
            return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

    # apply bias to activation tensor
    def _apply_bias(self, x):
        b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
        b = tf.cast(b, x.dtype)
        if len(x.shape) == 2:
            return x + b
        else:
            return x + tf.reshape(b, [1, -1, 1, 1])

    # convolutional layer
    def _conv2d(self, x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
        assert kernel >=1 and kernel % 2 == 1
        w = _get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, x.dtype)
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')

    # Fused upscale2d + conv2d
    def _upscale2d_conv2d(self, x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
        assert kernel >= 1 and kernel % 2 == 1
        w = _get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        w = tf.cast(w, x.dtype)
        os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
        return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

    # nearest-neighbor upscaling layer
    def _upscale2d(self, x, factor=2):
        assert isinstance(factor, int) and factor >= 1
        if factor == 1: return x
        with tf.variable_scope('Upscale2D'):
            s = x.shape
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
            return x

    # Fused conv2d + downscale2d
    def _conv2d_downscale2d(self, x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
        assert kernel >= 1 and kernel % 2 == 1
        w = _get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
        w = tf.cast(w, x.dtype)
        return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

    # box filter downscaling layer
    def _downscale2d(self, x, factor=2):
        assert isinstance(factor, int) and factor >= 1
        if factor == 1: return x
        with tf.variable_scope('Downscale2D'):
            ksize = [1, 1, factor, factor]
            return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')   # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

    # minibatch standard deviation
    def _minibatch_stddev_layer(self, x, group_size=4):
        with tf.variable_scope('MinibatchStddev'):
            group_size = tf.minimum(group_size, tf.shape(x)[0])
            s = x.shape
            y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])
            y = tf.cast(y, tf.float32)
            y-= tf.reduce_mean(y, axis=0, keepdims=True)
            y = tf.reduce_mean(tf.square(y), axis=0)
            y = tf.sqrt(y + 1e-8)
            y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)
            y = tf.cast(y, x.dtype)
            y = tf.tile(y, [group_size, 1, s[2], s[3]])
            return tf.concat([x, y], axis=1)

    def _lerp_clip(self, a, b, t):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)

    def _cset(self, cur_lambda, new_cond, new_lambda):
        return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

    def _lerp(self, a, b, t):
        return a + (b - a) * t


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
