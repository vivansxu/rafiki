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
            'D_repeats': IntegerKnob(1, 3)
            'minibatch_base': CategoricalKnob([4, 8, 16, 32]),
            'G_lrate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'D_lrate': FloatKnob(1e-5, 1e-1, is_exp=True),
            'lod_initial_resolution': FixedKnob(4),
        }

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self.num_gpus = 1
        np.random.seed(1000)
        print('Initializing TensorFlow...')

        tf_config = {}
        tf_config['graph_options.place_pruned_graph'] = True
        tf_config['gpu_options.allow_growth'] = True

        #TODO: any environment var needed?

        if tf.get_default_session() is None:
            tf.set_random_seed(np.random.randint(1 << 31))
            self._session = _create_session(config_dict=tf_config, force_as_default=True)

       
    def train(self, dataset_uri):
        D_repeats = self._knobs.get('D_repeats')
        _train_progressive_gan(num_gpus=self.num_gpus, D_repeats=D_repeats)

    def evaluate(self, dataset_uri):
        pass

    def predict(self, queries):
        pass

    def destroy(self):
        pass

    def dump_parameters(self):
        pass

    def load_parameters(self, params):
        pass

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
        num_gpus                = 1,
        G_smoothing             = 0.99,
        D_repeats               = 1,
        minibatch_repeats       = 4,
        reset_opt_for_new_lod   = True,
        total_kimg              = 12000,
        mirror_augment          = False,
        drange_net              = [-1, 1]):

        training_set = _load_dataset(data_dir=dataset_uri, {'tfrecord_dir': './'})
        with tf.device('/gpu:0'):
            print('Constructing networks...')
            G = Network('G', func='G_paper', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size)
            D = Network('D', func='D_paper', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size)
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

        config_G_opt = {'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-8}
        config_D_opt = {'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-8}
        G_opt = Optimizer(name='TrainG', learning_rate=lrate_in, **config_G_opt)
        D_opt = Optimizer(name='TrainD', learning_rate=lrate_in, **config_D_opt)

        for gpu in range(num_gpus):
            with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
                G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
                D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
                lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in)]
                reals_gpu = _process_reals(reals_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)
                labels_gpu = labels_split[gpu]
                
                with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
                    G_loss = G_wgan_acgan(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set, minibatch_size=minibatch_split, **config_G_loss)  # TODO: config_G_loss, config_D_loss
                with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
                    D_loss = D_wgangp_acgan(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals_gpu, labels=labels_gpu, **config_D_loss)
                G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
                D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)
        G_train_op = G_opt.apply_updates()
        D_train_op = D_opt.apply_updates()

        print('Training...')

        cur_nimg = 0
        prev_lod = -1
        config_sched = {}
        config_sched['minibatch_base'] = self._knobs.get('minibatch_base')
        config_sched['G_lrate'] = self._knobs.get('G_lrate')
        config_sched['D_lrate'] = self._knobs.get('D_lrate')
        config_sched['lod_initial_resolution'] = self._knobs.get('lod_initial_resolution')

        while cur_nimg < total_kimg * 1000:
            sched = TrainingSchedule(num_gpus, cur_nimg, training_set, **config_sched)
            training_set.configure(sched.minibatch, sched.lod)
            if reset_opt_for_new_lod:
                if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                    G_opt.reset_optimizer_state()
                    D_opt.reset_optimizer_state()
            prev_lod = sched.lod

            for repeat in range(minibatch_repeats):
                for _ in range(D_repeats):
                    tf.get_default_session().run([D_train_op, Gs_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
                    cur_nimg += sched.minibatch
                tf.get_default_session().run([G_train_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})

    def _process_reals(x, lod, mirror_augment, drange_data, drange_net):
        with tf.name_scope('ProcessReals'):
            with tf.name_scope('DynamicRange'):
                x = tf.cast(x, tf.float32)
                x = _adjust_dynamic_range(x, drange_data, drange_net)
            if mirror_augment:
                with tf.name_scope('MirrorAugment'):
                    s = tf.shape(x)
                    mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                    mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                    x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
            with tf.name_scope('FadeLOD'):
                s = tf.shape(x)
                y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
                y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
                y = tf.tile(y, [1, 1, 1, 2, 1, 2])
                y = tf.reshape(y, [-1, s[1], s[2], s[3]])
                x = x + (y - x) * (lod - tf.floor(lod))

            with tf.name_scope('UpscaleLOD'):
                s = tf.shape(x)
                factor = tf.cast(2 ** tf.floor(lod), tf.int32)
                x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
                x = tf.tile(x, [1, 1, 1, factor, 1, factor])
                x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
            return x

    def _adjust_dynamic_range(data, drange_in, drange_out):
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return data

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

class Optimizer:
    def __init__(
        self,
        name                = 'Train',
        tf_optimizer        = 'tf.train.AdamOptimizer',
        learning_rate       = 0.001,
        use_loss_scaling    = False,
        loss_scaling_init   = 64.0,
        loss_scaling_inc    = 0.0005,
        loss_scaling_dec    = 1.0,
        **kwargs):

        self.name = name
        self.learning_rate = tf.convert_to_tensor(learning_rate)
        self.id = self.name.replace('/', '.')
        self.scope = tf.get_default_graph().unique_name(self.id)
        self.optimizer_class = import_obj(tf_optimizer)
        self.optimizer_kwargs = dict(kwargs)
        self.use_loss_scaling = use_loss_scaling
        self.loss_scaling_init = loss_scaling_init
        self.loss_scaling_inc = loss_scaling_inc
        self.loss_scaling_dec = loss_scaling_dec
        self._grad_shapes = None
        self._dev_opt = OrderedDict()
        self._dev_grads = OrderedDict()
        self._dev_ls_var = OrderedDict()
        self._updates_applied = False

    def register_gradients(self, loss, vars):
        assert not self._updates_applied

        if isinstance(vars, dict):
            vars = list(vars.values())
        assert isinstance(vars, list) and len(vars) >= 1
        assert all((isinstance(expr, tf.Tensor) or isinstance(expr, tf.Variable) or isinstance(expr, tf.Operation)) for expr in vars + [loss])
        if self._grad_shapes is None:
            self._grad_shapes = [[dim.value for dim in var.shape] for var in vars]
        assert len(vars) == len(self._grad_shapes)
        assert all([dim.value for dim in var.shape] == var_shape for var, var_shape in zip(vars, self._grad_shapes))
        dev = loss.device
        assert all(var.device == dev for var in vars)

        with tf.name_scope(self.id + '_grad'), tf.device(dev):
            if dev not in self._dev_opt:
                opt_name = self.scope.replace('/', '_') + '_opt%d' % len(self._dev_opt)
                self._dev_opt[dev] = self.optimizer_class(name=opt_name, learning_rate=self.learning_rate, **self.optimizer_kwargs)
                self._dev_grads[dev] = []
            loss = self.apply_loss_scaling(tf.cast(loss, tf.float32))
            grads = self._dev_opt[dev].compute_gradients(loss, vars, gate_gradients=tf.train.Optimizer.GATE_NONE)
            grads = [(g, v) if g is not None else (tf.zeros_like(v), v) for g, v in grads]
            self._dev_grads[dev].append(grads)

    def apply_updates(self):
        assert not self._updates_applied
        self._updates_applied = True
        devices = list(self._dev_grads.keys())
        total_grads = sum(len(grads) for grads in self._dev_grads.values())
        assert len(devices) >= 1 and total_grads >= 1
        ops = []
        with tf.name_scope(self.scope + '/'):
            dev_grads = OrderedDict()
            for dev_idx, dev in enumerate(devices):
                with tf.name_scope('ProcessGrads%d' % dev_idx), tf.device(dev):
                    sums = []
                    for gv in zip(*self._dev_grads[dev]):
                        assert all(v is gv[0][1] for g, v in gv)
                        g = [tf.cast(g, tf.float32) for g, v in gv]
                        g = g[0] if len(g) == 1 else tf.add_n(g)
                        sums.append((g, gv[0][1]))
                    dev_grads[dev] = sums
            
            if len(devices) > 1:
                with tf.name_scope('SumAcrossGPUs'), tf.device(None):
                    for var_idx, grad_shape in enumerate(self._grad_shapes):
                        g = [dev_grads[dev][var_idx][0] for dev in devices]
                        if np.prod(grad_shape): # nccl does not support zero-sized tensors
                            g = tf.contrib.nccl.all_sum(g)
                        for dev, gg in zip(devices, g):
                            dev_grads[dev][var_idx] = (gg, dev_grads[dev][var_idx][1])

            for dev_idx, (dev, grads) in enumerate(dev_grads.items()):
                with tf.name_scope('ApplyGrads%d' % dev_idx), tf.device(dev):
                    if self.use_loss_scaling or total_grads > 1:
                        with tf.name_scope('Scale'):
                            coef = tf.constant(np.float32(1.0 / total_grads), name='coef')
                            coef = self.undo_loss_scaling(coef)
                            grads = [(g * coef, v) for g, v in grads]
                    with tf.name_scope('CheckOverflow'):
                        grad_ok = tf.reduce_all(tf.stack([tf.reduce_all(tf.is_finite(g)) for g, v in grads]))

                    with tf.name_scope('UpdateWeights'):
                        opt = self._dev_opt[dev]
                        ls_var = self.get_loss_scaling_var(dev)
                        if not self.use_loss_scaling:
                            ops.append(tf.cond(grad_ok, lambda: opt.apply_gradients(grads), tf.no_op))
                        else:
                            ops.append(tf.cond(grad_ok,
                                lambda: tf.group(tf.assign_add(ls_var, self.loss_scaling_inc), opt.apply_gradients(grads)),
                                lambda: tf.group(tf.assign_sub(ls_var, self.loss_scaling_dec))))

                    if dev == devices[-1]:
                        with tf.name_scope('Statistics'):
                            ops.append(_autosummary(self.id + '/learning_rate', self.learning_rate))
                            ops.append(_autosummary(self.id + '/overflow_frequency', tf.where(grad_ok, 0, 1)))
                            if self.use_loss_scaling:
                                ops.append(_autosummary(self.id + '/loss_scaling_log2', ls_var))

            self.reset_optimizer_state()
            init_uninited_vars(list(self._dev_ls_var.values()))
            return tf.group(*ops, name='TrainingOp')

    def reset_optimizer_state(self):
        tf.get_default_session().run([var.initializer for opt in self._dev_opt.values() for var in opt.variables()])

    def get_loss_scaling_var(self, device):
        if not self.use_loss_scaling:
            return None
        if device not in self._dev_ls_var:
            with tf.name_scope(self.scope + '/LossScalingVars/'), tf.control_dependencies(None):
                self._dev_ls_var[device] = tf.Variable(np.float32(self.loss_scaling_init), name='loss_scaling_var')
        return self._dev_ls_var[device]

    def apply_loss_scaling(self, value):
        assert isinstance(value, tf.Tensor) or isinstance(value, tf.Variable) or isinstance(value, tf.Operation)
        if not self.use_loss_scaling:
            return value
        return value * exp2(self.get_loss_scaling_var(value.device))

    def undo_loss_scaling(self, value):
        assert isinstance(value, tf.Tensor) or isinstance(value, tf.Variable) or isinstance(value, tf.Operation)
        if not self.use_loss_scaling:
            return value
        return value * exp2(-self.get_loss_scaling_var(value.device))
        
class TrainingSchedule:
    def __init__(self,
        num_gpus                = 1,
        cur_nimg,
        training_set,
        lod_initial_resolution  = 4,
        lod_training_kimg       = 600,
        lod_transition_kimg     = 600,
        minibatch_base          = 16,
        minibatch_dict          = {},
        max_minibatch_per_gpu   = {256: 16, 512: 8, 1024: 4},
        G_lrate                 = 0.001,
        #G_lrate_dict            = {},
        D_lrate                 = 0.001,
        #D_lrate_dict            = {},
        ):

        if minibatch_base == 4:
            minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
        elif minibatch_base == 8:
            minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
        elif minibatch_base == 16:
            minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
        elif minibatch_base == 32:
            minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}

        self.kimg = cur_nimg / 1000.0
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = self.kimg - phase_idx * phase_dur

        self.lod = training_set.resolution_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(self.lod)))

        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * num_gpus)
        
        #self.G_lrate = G_lrate_dict.get(self.resolution, G_lrate_base)
        #self.D_lrate = D_lrate_dict.get(self.resolution, D_lrate_base)
        self.G_lrate = G_lrate
        self.D_lrate = D_lrate

def G_wgan_acgan(G, D, opt, training_set, minibatch_size, cond_weight = 1.0):

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight

    return loss

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda = 10.0,
    wgan_epsilon = 0.001,
    wgan_target = 1.0,
    cond_weight = 1.0):

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = _autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = _autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tf.cast(reals, fake_images_out.dtype) + (fake_images_out - tf.cast(reals, fake_images_out.dtype)) * mixing_factors
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = _autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = _autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = _autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = _autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = _autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def exp2(x):
    with tf.name_scope('Exp2'):
        return tf.exp(x * np.float32(np.log(2.0)))

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

def init_uninited_vars(vars=None):
    if vars is None: vars = tf.global_variables()
    test_vars = []; test_ops = []
    with tf.control_dependencies(None): # ignore surrounding control_dependencies
        for var in vars:
            assert is_tf_expression(var)
            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(':0', '/IsVariableInitialized:0'))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)
                with absolute_name_scope(var.name.split(':')[0]):
                    test_ops.append(tf.is_variable_initialized(var))
    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    run([var.initializer for var in init_vars]

def import_module(module_or_obj_name):
    parts = module_or_obj_name.split('.')
    parts[0] = {'np': 'numpy', 'tf': 'tensorflow'}.get(parts[0], parts[0])
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module('.'.join(parts[:i]))
            relative_obj_name = '.'.join(parts[i:])
            return module, relative_obj_name
        except ImportError:
            pass
    raise ImportError(module_or_obj_name)

def find_obj_in_module(module, relative_obj_name):
    obj = module
    for part in relative_obj_name.split('.'):
        obj = getattr(obj, part)
    return obj

def import_obj(obj_name):
    module, relative_obj_name = import_module(obj_name)
    return find_obj_in_module(module, relative_obj_name)

_autosummary_vars = OrderedDict() # name => [var, ...]
_autosummary_immediate = OrderedDict() # name => update_op, update_value
_autosummary_finalized = False

def _autosummary(name, value):
    id = name.replace('/', '_')
    if isinstance(value, tf.Tensor) or isinstance(value, tf.Variable) or isinstance(value, tf.Operation):
        with tf.name_scope('summary_' + id), tf.device(value.device):
            update_op = _create_autosummary_var(name, value)
            with tf.control_dependencies([update_op]):
                return tf.identity(value)
    else: # python scalar or numpy array
        if name not in _autosummary_immediate:
            with tf.name_scope('Autosummary/' + id + '/'), tf.device(None), tf.control_dependencies(None):
                update_value = tf.placeholder(tf.float32)
                update_op = _create_autosummary_var(name, update_value)
                _autosummary_immediate[name] = update_op, update_value
        update_op, update_value = _autosummary_immediate[name]
        tf.get_default_session().run(update_op, {update_value: np.float32(value)})
        return value

def _create_autosummary_var(name, value_expr):
    assert not _autosummary_finalized
    v = tf.cast(value_expr, tf.float32)
    if v.shape.ndims is 0:
        v = [v, np.float32(1.0)]
    elif v.shape.ndims is 1:
        v = [tf.reduce_sum(v), tf.cast(tf.shape(v)[0], tf.float32)]
    else:
        v = [tf.reduce_sum(v), tf.reduce_prod(tf.cast(tf.shape(v), tf.float32))]
    v = tf.cond(tf.is_finite(v[0]), lambda: tf.stack(v), lambda: tf.zeros(2))
    with tf.control_dependencies(None):
        var = tf.Variable(tf.zeros(2)) # [numerator, denominator]
    update_op = tf.cond(tf.is_variable_initialized(var), lambda: tf.assign_add(var, v), lambda: tf.assign(var, v))
    if name in _autosummary_vars:
        _autosummary_vars[name].append(var)
    else:
        _autosummary_vars[name] = [var]
    return update_op

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
