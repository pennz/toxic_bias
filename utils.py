import os
import pickle
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

BIN_FOLDER = './'

def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()

def dump_obj(obj, filename, fullpath=False, force=False):
    if not fullpath:
        path = BIN_FOLDER+filename
    else:
        path = filename
    if not force and os.path.isfile(path):
        print(f"{path} already existed, not dumping")
    else:
        print(f"Overwrite {path}!")
        pickle.dump(obj, open(path, 'wb'))

def get_obj_and_save(filename, default=None, fullpath=False):
    if not fullpath:
        path = BIN_FOLDER+filename
    else:
        path = filename
    if os.path.isfile(path):
        logger.debug("load :"+filename)
        return pickle.load(open(path, 'rb'))
    else:
        if default is not None:
            logger.debug("dump :"+filename)
            dump_obj(default, filename)
        return default

def file_exist(filename, fullpath=False):
    if not fullpath:
        path = BIN_FOLDER+filename
    else:
        path = filename
    return os.path.isfile(path)

def binary_crossentropy_with_focal_seasoned(y_true, logit_pred, beta=0., gamma=1., alpha=0.5, custom_weights_in_y_true=True):  # 0.5 means no rebalance
    """
    :param alpha:weight for positive classes **loss**. default to 1- true positive cnts / all cnts, alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practiceαmay be set by inverse class freqency or hyperparameter.
    :param custom_weights_in_y_true:
    :return:
    """
    balanced = gamma*logit_pred + beta
    y_pred = math_ops.sigmoid(balanced)
    return binary_crossentropy_with_focal(y_true, y_pred, gamma=0, alpha=alpha, custom_weights_in_y_true=custom_weights_in_y_true)  # only use gamma in this layer, easier to split out factor


def binary_crossentropy_with_focal(y_true, y_pred, gamma=1., alpha=0.5, custom_weights_in_y_true=True):  # 0.5 means no rebalance
    """
    https://arxiv.org/pdf/1708.02002.pdf

    $$ FL(p_t) = -(1-p_t)^{\gamma}log(p_t) $$
    $$ p_t=p\: if\: y=1$$
    $$ p_t=1-p\: otherwise$$

    :param y_true:
    :param y_pred:
    :param gamma: make easier ones weights down
    :param alpha: weight for positive classes. default to 1- true positive cnts / all cnts, alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practiceαmay be set by inverse class freqency or hyperparameter.
    :return:
    """
    # assert 0 <= alpha <= 1 and gamma >= 0
    # hyper parameters, just use the one for binary?
    # alpha = 1. # maybe smaller one can help, as multi-class will make the error larger
    # gamma = 1.5 # for our problem, try different gamma

    # for binary_crossentropy, the implementation is in  tensorflow/tensorflow/python/keras/backend.py
    #       bce = target * alpha* (1-output+epsilon())**gamma * math_ops.log(output + epsilon())
    #       bce += (1 - target) *(1-alpha)* (output+epsilon())**gamma * math_ops.log(1 - output + epsilon())
    # return -bce # binary cross entropy
    eps = tf.keras.backend.epsilon()

    if custom_weights_in_y_true:
        custom_weights = y_true[:, 1:2]
        y_true = y_true[:, :1]

    if 1. - eps <= gamma <= 1. + eps:
        bce = alpha * math_ops.multiply(1. - y_pred, math_ops.multiply(y_true, math_ops.log(y_pred + eps)))
        bce += (1 - alpha) * math_ops.multiply(y_pred,
                                               math_ops.multiply((1. - y_true), math_ops.log(1. - y_pred + eps)))
    elif 0. - eps <= gamma <= 0. + eps:
        bce = alpha * math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        bce += (1 - alpha) * math_ops.multiply((1. - y_true), math_ops.log(1. - y_pred + eps))
    else:
        gamma_tensor = tf.broadcast_to(tf.constant(gamma), tf.shape(input=y_pred))
        bce = alpha * math_ops.multiply(math_ops.pow(1. - y_pred, gamma_tensor),
                                        math_ops.multiply(y_true, math_ops.log(y_pred + eps)))
        bce += (1 - alpha) * math_ops.multiply(math_ops.pow(y_pred, gamma_tensor),
                                               math_ops.multiply((1. - y_true), math_ops.log(1. - y_pred + eps)))

    if custom_weights_in_y_true:
        return math_ops.multiply(-bce, custom_weights)
    else:
        return -bce

def reinitLayers(model):
    session = K.get_session()
    for layer in model.layers:
        #if isinstance(layer, keras.engine.topology.Container):
        if isinstance(layer, tf.keras.Model):
            reinitLayers(layer)
            continue
        print("LAYER::", layer.name)
        if layer.trainable == False:
            continue
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg,'initializer'):  # not work for layer wrapper, like Bidirectional
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
                print('reinitializing layer {}.{}'.format(layer.name, v))

from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K

class AttentionRaffel(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        :param step_dim: feature vector length
        :param W_regularizer:
        :param b_regularizer:
        :param W_constraint:
        :param b_constraint:
        :param bias:
        :param kwargs:
        """
        super(AttentionRaffel, self).__init__(**kwargs)
        self.supports_masking = True
        self.init = 'glorot_uniform'

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

    def get_config(self):
        config = {
            'step_dim':
                self.step_dim,
            'bias':
                self.bias,
            'W_regularizer':
                regularizers.serialize(self.W_regularizer),
            'b_regularizer':
                regularizers.serialize(self.b_regularizer),
            'W_constraint':
                constraints.serialize(self.W_constraint),
            'b_constraint':
                constraints.serialize(self.b_constraint),
        }
        base_config = super(AttentionRaffel, self).get_config()
        if 'cell' in base_config: del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # Input shape 3D tensor with shape: `(samples, steps, features)`.
        # one step is means one bidirection?
        assert len(input_shape) == 3

        self.W = self.add_weight('{}_W'.format(self.name),
                                 (int(input_shape[-1]),),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]  # features dimention of input

        if self.bias:
            self.b = self.add_weight('{}_b'.format(self.name),
                                     (int(input_shape[1]),),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # more like the alignment model, which scores how the inputs around position j and the output
        # at position i match
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)  # activation

        # softmax
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a  # context vector c_i (or for this, only one c_i)
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim

class NBatchProgBarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, count_mode='samples', stateful_metrics=None, display_per_batches=1000, verbose=1,
                 early_stop=False, patience_displays=0, epsilon=1e-7, batch_size=1024):
        super(NBatchProgBarLogger, self).__init__(count_mode, stateful_metrics)
        self.display_per_batches = 1 if display_per_batches < 1 else display_per_batches
        self.step_idx = 0  # across epochs
        self.display_idx = 0  # across epochs
        self.verbose = verbose

        self.early_stop = early_stop  # better way is subclass EearlyStopping callback.
        self.patience_displays = patience_displays
        self.losses = np.empty(patience_displays, dtype=np.float32)
        self.losses_sum_display = 0
        self.epsilon = epsilon
        self.stopped_step = 0
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen` calculation.
        num_steps = logs.get('num_steps', 1)
        if self.use_steps:
            self.seen += num_steps
        else:
            self.seen += batch_size * num_steps

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        self.step_idx += 1
        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.early_stop:
            loss = logs.get('loss')  # only record for this batch, not the display. Should work
            self.losses_sum_display += loss

        if self.step_idx % self.display_per_batches == 0:
            if self.verbose and self.seen < self.target:
                self.progbar.update(self.seen, self.log_values)

            if self.early_stop:
                avg_loss_per_display = self.losses_sum_display / self.display_per_batches
                self.losses_sum_display = 0  # clear mannually...
                self.losses[self.display_idx % self.patience_displays] = avg_loss_per_display
                # but it still SGD, variance still, it just smaller by factor of display_per_batches
                display_info_start_step = self.step_idx - self.display_per_batches + 1
                print(
                    f'\nmean: {avg_loss_per_display}, Step {display_info_start_step }({display_info_start_step*self.batch_size}) to {self.step_idx}({self.step_idx*self.batch_size}) for {self.display_idx}th display step')

                self.display_idx += 1  # used in index, so +1 later
                if self.display_idx >= self.patience_displays:
                    std = np.std(
                        self.losses)  # as SGD, always variance, so not a good way, need to learn from early stopping
                    std_start_step = self.step_idx - self.display_per_batches * self.patience_displays + 1
                    print(f'mean: {np.mean(self.losses)}, std:{std} for Step {std_start_step}({std_start_step*self.batch_size}) to {self.step_idx}({self.step_idx*self.batch_size}) for {self.display_idx}th display steps')
                    if std < self.epsilon:
                        self.stopped_step = self.step_idx
                        self.model.stop_training = True
                        print(
                            f'Early Stop criterion met: std is {std} at Step {self.step_idx} for {self.display_idx}th display steps')

    def on_train_end(self, logs=None):
        if self.stopped_step > 0 and self.verbose > 0:
            print('Step %05d: early stopping' % (self.stopped_step + 1))

class KaggleKernel:
    def __init__(self):
        pass

    def build_model(self):
        pass

    def set_loss(self):
        pass

    def set_model(self):
        pass

    def set_metrics(self):
        pass

    def prepare_train_data(self):
        pass

    def prepare_dev_data(self):
        pass

    def prepare_test_data(self):
        pass

    def save_result(self):
        pass
