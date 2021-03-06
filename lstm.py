import argparse
import copy
import gc
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.core.debugger import set_trace
from sklearn.model_selection import KFold
from tensorflow import Graph, Session
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ProgbarLogger)
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Bidirectional, CuDNNLSTM, Dense,
                                     Embedding, GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, Input, Lambda, Layer,
                                     PReLU, SpatialDropout1D, add, concatenate)
from tensorflow.keras.metrics import (binary_crossentropy, mean_absolute_error,
                                      mean_squared_error)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import math_ops

import data_prepare as d
import utils

# from gradient_reversal_keras_tf.flipGradientTF import GradientReversal


NUM_MODELS = 2  # might be helpful but...

# BATCH_SIZE = 2048 * 2  # for cloud server runing
BATCH_SIZE = 1024 // 2  # for cloud server runing
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
RES_DENSE_HIDDEN_UNITS = 5

EPOCHS = (
    7  # 4 seems good for current setting, more training will help for the final score?
)

DEBUG_TRACE = False
# all for debug
preds = None
kernel = None
X, y, idx_train_background, idx_val_background = None, None, None, None
y_res_pred = None

# lambda e: 5e-3 * (0.5 ** e),
STARTER_LEARNING_RATE = 1e-3  # as the BCE we adopted...
LEARNING_RATE_DECAY_PER_EPOCH = 0.5

IDENTITY_RUN = True
TARGET_RUN = "lstm"
TARGET_RUN_READ_RESULT = False
RESTART_TRAIN = False
RESTART_TRAIN_RES = True
RESTART_TRAIN_ID = False

NO_AUX = False
Y_TRAIN_BIN = False  # with True, slightly worse

FOCAL_LOSS = True

FOCAL_LOSS_GAMMA = 0.0
FOCAL_LOSS_GAMMA_NEG_POS = 0.25
FOCAL_LOSS_BETA_NEG_POS = 1.0
# ALPHA = 0.91
ALPHA = 0.5
PRD_ONLY = True  # will not train the model
NOT_PRD = True
FINAL_SUBMIT = False
FINAL_DEBUG = True
if FINAL_SUBMIT:
    TARGET_RUN = "lstm"
    EPOCHS = 6
    PRD_ONLY = False  # not training
    NOT_PRD = False  # prd submission test
    RESTART_TRAIN = True
    FOCAL_LOSS_GAMMA = 0.0

    DEBUG_TRACE = False

    if FINAL_DEBUG:
        EPOCHS = 4
        DEBUG_TRACE = False
        PRD_ONLY = False  # not training if True
        RESTART_TRAIN = False
        NOT_PRD = True  # prd submission test
        NUM_MODELS = 1  # might timeout, so back to 2
        STARTER_LEARNING_RATE = 1e-3
        Y_TRAIN_BIN = False


# Credits for https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043


class AttentionRaffel(Layer):
    def __init__(
        self,
        step_dim=d.MAX_LEN,
        W_regularizer=None,
        b_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        bias=True,
        **kwargs,
    ):
        """
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        :param step_dim:
        :param W_regularizer:
        :param b_regularizer:
        :param W_constraint:
        :param b_constraint:
        :param bias:
        :param kwargs:
        """
        super(AttentionRaffel, self).__init__(**kwargs)
        self.supports_masking = True
        self.init = "glorot_uniform"

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

    def get_config(self):
        config = {
            "step_dim": self.step_dim,
            "bias": self.bias,
            "W_regularizer": regularizers.serialize(self.W_regularizer),
            "b_regularizer": regularizers.serialize(self.b_regularizer),
            "W_constraint": constraints.serialize(self.W_constraint),
            "b_constraint": constraints.serialize(self.b_constraint),
        }
        base_config = super(AttentionRaffel, self).get_config()
        if "cell" in base_config:
            del base_config["cell"]
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # Input shape 3D tensor with shape: `(samples, steps, features)`.
        # one step is means one bidirection?
        assert len(input_shape) == 3

        self.W = self.add_weight(
            "{}_W".format(self.name),
            (int(input_shape[-1]),),
            initializer=self.init,
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        self.features_dim = input_shape[-1]  # features dimention of input

        if self.bias:
            self.b = self.add_weight(
                "{}_b".format(self.name),
                (int(input_shape[1]),),
                initializer="zero",
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
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

        eij = K.reshape(
            K.dot(
                K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))
            ),
            (-1, step_dim),
        )

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
        # context vector c_i (or for this, only one c_i)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


def identity_model(features, labels, mode, params):
    # words = Input(shape=(None,))
    embedding_matrix = params["embedding_matrix"]
    identity_out_num = params["identity_out_num"]

    words = features  # input layer, so features need to be tensor!!!

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(
        words
    )
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x),])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation="relu")(hidden)])

    # Output Layer
    result = Dense(identity_out_num, activation="sigmoid")(hidden)

    # Implement training, evaluation, and prediction
    # Compute predictions.
    # predicted_classes = tf.argmax(input=logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # predictions = {
        #    'class_ids': predicted_classes[:, tf.newaxis],
        #    'probabilities': tf.nn.softmax(logits),
        #    'logits': logits,
        # }
        return tf.estimator.EstimatorSpec(mode, predictions=result.out)

    # Compute loss.
    loss = tf.keras.losses.binary_crossentropy(
        labels, result
    )  # todo put it together, fine?

    # Compute evaluation metrics.
    # m = tf.keras.metrics.SparseCategoricalAccuracy()
    # m.update_state(labels, logits)
    # accuracy = m.result()
    # metrics = {'accuracy': accuracy}
    # tf.compat.v1.summary.scalar('accuracy', accuracy)
    binary_accuracy = tf.keras.metrics.binary_accuracy(labels, result)

    metrics = {"accuracy": binary_accuracy}
    tf.compat.v1.summary.scalar("accuracy", binary_accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op. # will be called by the estimator
    assert mode == tf.estimator.ModeKeys.TRAIN

    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate'])
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate=params["learning_rate"],
            global_step=tf.compat.v1.train.get_global_step(),
            decay_steps=params["decay_steps"],
            staircase=True,
            decay_rate=0.5,
        )
    )
    train_op = optimizer.minimize(
        loss, global_step=tf.compat.v1.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class NBatchProgBarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(
        self,
        count_mode="samples",
        stateful_metrics=None,
        display_per_batches=1000,
        verbose=1,
        early_stop=False,
        patience_displays=0,
        epsilon=1e-7,
    ):
        super(NBatchProgBarLogger, self).__init__(count_mode, stateful_metrics)
        self.display_per_batches = 1 if display_per_batches < 1 else display_per_batches
        self.step_idx = 0  # across epochs
        self.display_idx = 0  # across epochs
        self.verbose = verbose

        # better way is subclass EearlyStopping callback.
        self.early_stop = early_stop
        self.patience_displays = patience_displays
        self.losses = np.empty(patience_displays, dtype=np.float32)
        self.losses_sum_display = 0
        self.epsilon = epsilon
        self.stopped_step = 0

    def on_train_begin(self, logs=None):
        self.epochs = self.params["epochs"]

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get("size", 0)
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen` calculation.
        num_steps = logs.get("num_steps", 1)
        if self.use_steps:
            self.seen += num_steps
        else:
            self.seen += batch_size * num_steps

        for k in self.params["metrics"]:
            if k in logs:
                self.log_values.append((k, logs[k]))

        self.step_idx += 1
        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.early_stop:
            # only record for this batch, not the display. Should work
            loss = logs.get("loss")
            self.losses_sum_display += loss

        if self.step_idx % self.display_per_batches == 0:
            if self.verbose and self.seen < self.target:
                self.progbar.update(self.seen, self.log_values)

            if self.early_stop:
                avg_loss_per_display = (
                    self.losses_sum_display / self.display_per_batches
                )
                self.losses_sum_display = 0  # clear mannually...
                self.losses[
                    self.display_idx % self.patience_displays
                ] = avg_loss_per_display
                # but it still SGD, variance still, it just smaller by factor of display_per_batches
                display_info_start_step = self.step_idx - self.display_per_batches + 1
                print(
                    f"\nmean: {avg_loss_per_display}, Step {display_info_start_step }({display_info_start_step*BATCH_SIZE}) to {self.step_idx}({self.step_idx*BATCH_SIZE}) for {self.display_idx}th display step"
                )

                self.display_idx += 1  # used in index, so +1 later
                if self.display_idx >= self.patience_displays:
                    std = np.std(
                        self.losses
                    )  # as SGD, always variance, so not a good way, need to learn from early stopping
                    std_start_step = (
                        self.step_idx
                        - self.display_per_batches * self.patience_displays
                        + 1
                    )
                    print(
                        f"mean: {np.mean(self.losses)}, std:{std} for Step {std_start_step}({std_start_step*BATCH_SIZE}) to {self.step_idx}({self.step_idx*BATCH_SIZE}) for {self.display_idx}th display steps"
                    )
                    if std < self.epsilon:
                        self.stopped_step = self.step_idx
                        self.model.stop_training = True
                        print(
                            f"Early Stop criterion met: std is {std} at Step {self.step_idx} for {self.display_idx}th display steps"
                        )

    def on_train_end(self, logs=None):
        if self.stopped_step > 0 and self.verbose > 0:
            print("Step %05d: early stopping" % (self.stopped_step + 1))


# refer to https://github.com/keras-team/keras/issues/341


def reinitLayers(model):
    session = K.get_session()
    for layer in model.layers:
        # if isinstance(layer, keras.engine.topology.Container):
        if isinstance(layer, tf.keras.Model):
            reinitLayers(layer)
            continue
        print("LAYER::", layer.name)
        if layer.trainable == False:
            continue
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(
                v_arg, "initializer"
            ):  # not work for layer wrapper, like Bidirectional
                initializer_method = getattr(v_arg, "initializer")
                initializer_method.run(session=session)
                print("reinitializing layer {}.{}".format(layer.name, v))


class KaggleKernel:
    def __init__(self, action=None):
        self.model = None
        self.emb = None
        self.train_X = None
        self.train_X_all = None
        self.train_y_all = None
        self.train_y = None
        self.train_y_aux = None
        self.train_y_aux_all = None
        self.train_y_identity = None
        self.train_X_identity = None
        # self.to_predict_X = None
        self.embedding_matrix = None
        self.embedding_layer = None
        self.identity_idx = None
        self.id_used_in_train = False
        self.id_validate_df = None

        self.judge = None  # for auc metrics

        self.oof_preds = None
        self.load_data(action)

    def load_data(self, action):
        if self.emb is None:
            self.emb = d.EmbeddingHandler()
        self.emb.read_train_test_df(train_only=False)
        self.train_df = self.emb.train_df

        if PRD_ONLY or TARGET_RUN_READ_RESULT or ANA_RESULT:
            pass
        else:  # keep record training parameters
            self.emb.dump_obj(
                f"Start run with FL_{FOCAL_LOSS}_{FOCAL_LOSS_GAMMA}_{ALPHA} lr {STARTER_LEARNING_RATE}, decay {LEARNING_RATE_DECAY_PER_EPOCH}, \
BS {BATCH_SIZE}, NO_ID_IN_TRAIN {EXCLUDE_IDENTITY_IN_TRAIN}, EPOCHS {EPOCHS}, Y_TRAIN_BIN {Y_TRAIN_BIN}",
                "run_info.txt",
                force=True,
            )

        (
            self.train_X_all,
            self.train_y_all,
            self.train_y_aux_all,
            self.to_predict_X_all,
            self.embedding_matrix,
        ) = self.emb.data_prepare(action)
        # picked ones are binarized
        self.train_y_aux_all = self.train_df[d.AUX_COLUMNS].values
        if Y_TRAIN_BIN:
            self.train_y_float_backup = self.train_y_all
            self.train_y_all = np.where(self.train_y_all >= 0.5, True, False)
        # if action is not None:
        #    if action == TRAIN_DATA_EXCLUDE_IDENDITY_ONES:
        self.load_identity_data_idx()
        mask = np.ones(len(self.train_X_all), np.bool)
        # identities data excluded first ( will add some back later)
        mask[self.identity_idx] = 0

        # need get more 80, 000 normal ones without identities, 40,0000 %40 with identities, 4*0.4/12, 0.4/3 13%
        add_no_identity_to_val = self.train_df[mask].sample(
            n=int(8e4), random_state=2019
        )
        add_no_identity_to_val_idx = add_no_identity_to_val.index
        mask[add_no_identity_to_val_idx] = 0  # exclude from train, add to val

        self.train_mask = mask

        self.train_X = self.train_X_all[mask]
        self.train_y = self.train_y_all[mask]
        self.train_y_aux = self.train_y_aux_all[mask]
        logger.debug("Train data no identity ones now")

        try:
            if self.emb.do_emb_matrix_preparation:
                exit(0)  # saved and can exit
        except:
            logger.warning(
                "Prepare emb for embedding error, so we might already have process file and load data, and we continue"
            )

    @staticmethod
    def bin_prd_clsf_info_neg(y_true, y_pred, threshold=0.5, N_MORE=True, epsilon=1e-7):
        """
        refer to this: https://stats.stackexchange.com/questions/49579/balanced-accuracy-vs-f-1-score

        Both F1 and b_acc are metrics for classifier evaluation, that (to some extent) handle class imbalance. Depending
         of which of the two classes (N or P) outnumbers the other, each metric is outperforms the other.

        1) If N >> P, f1 is a better.

        2) If P >> N, b_acc is better.

        For code: refer to this: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/70841

        :param y_true:
        :param y_pred:
        :param threshold:
        :return: accuracy, f1 for this batch... not the global one, we need to be careful!!
        """
        # if FOCAL_LOSS_GAMMA == 2.0:
        #    threshold = 0.57
        # elif FOCAL_LOSS_GAMMA == 1.0:
        #    threshold = (0.53 + (
        #                0.722 - 0.097)) / 2  # (by...reading the test result..., found it changes every training... so useless)
        threshold = math_ops.cast(threshold, y_pred.dtype)
        y_pred_b = math_ops.cast(y_pred > threshold, y_pred.dtype)
        y_true_b = math_ops.cast(y_true > threshold, y_pred.dtype)

        # ground_pos = math_ops.reduce_sum(y_true) + epsilon
        # correct_pos = math_ops.reduce_sum(math_ops.multiply(y_true, y_pred)) + epsilon
        # predict_pos = math_ops.reduce_sum(y_pred) + epsilon
        true_cnt = math_ops.reduce_sum(y_true_b) + epsilon
        false_cnt = math_ops.reduce_sum(1 - y_true_b) + epsilon
        pred_true_cnt = math_ops.reduce_sum(y_pred_b) + epsilon
        pred_false_cnt = math_ops.reduce_sum(1 - y_pred_b) + epsilon
        # true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
        # false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

        # true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
        # false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

        # tp_mean_scaled = math_ops.cast(true_predict_mean*100, tf.int8)
        # tp_mean_scaled = math_ops.cast(tp_mean_scaled, tf.float32)
        # precision = math_ops.div(correct_pos, predict_pos)
        # recall = math_ops.div(correct_pos, ground_pos)

        # if N_MORE:
        #    m = (2 * recall * precision) / (precision + recall)
        # else:
        #    # m = (sensitivity + specificity)/2 # balanced accuracy
        #    raise NotImplementedError("Balanced accuracy metric is not implemented")

        return (pred_false_cnt - false_cnt) / false_cnt  # (batchsize 1024)

    @staticmethod
    def bin_prd_clsf_info_pos(y_true, y_pred, threshold=0.5, N_MORE=True, epsilon=1e-7):
        """
        refer to this: https://stats.stackexchange.com/questions/49579/balanced-accuracy-vs-f-1-score

        Both F1 and b_acc are metrics for classifier evaluation, that (to some extent) handle class imbalance. Depending
         of which of the two classes (N or P) outnumbers the other, each metric is outperforms the other.

        1) If N >> P, f1 is a better.

        2) If P >> N, b_acc is better.

        For code: refer to this: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/70841

        :param y_true:
        :param y_pred:
        :param threshold:
        :return: accuracy, f1 for this batch... not the global one, we need to be careful!!
        """
        # if FOCAL_LOSS_GAMMA == 2.0:
        #    threshold = 0.57
        # elif FOCAL_LOSS_GAMMA == 1.0:
        #    threshold = (0.53 + (
        #                0.722 - 0.097)) / 2  # (by...reading the test result..., found it changes every training... so useless)
        threshold = math_ops.cast(threshold, y_pred.dtype)
        y_pred_b = math_ops.cast(y_pred > threshold, y_pred.dtype)
        y_true_b = math_ops.cast(y_true > threshold, y_pred.dtype)

        # ground_pos = math_ops.reduce_sum(y_true) + epsilon
        # correct_pos = math_ops.reduce_sum(math_ops.multiply(y_true, y_pred)) + epsilon
        # predict_pos = math_ops.reduce_sum(y_pred) + epsilon
        true_cnt = math_ops.reduce_sum(y_true_b) + epsilon
        false_cnt = math_ops.reduce_sum(1 - y_true_b) + epsilon
        pred_true_cnt = math_ops.reduce_sum(y_pred_b) + epsilon
        pred_false_cnt = math_ops.reduce_sum(1 - y_pred_b) + epsilon
        # true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
        # false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

        # true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
        # false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

        # tp_mean_scaled = math_ops.cast(true_predict_mean*100, tf.int8)
        # tp_mean_scaled = math_ops.cast(tp_mean_scaled, tf.float32)
        # precision = math_ops.div(correct_pos, predict_pos)
        # recall = math_ops.div(correct_pos, ground_pos)

        # if N_MORE:
        #    m = (2 * recall * precision) / (precision + recall)
        # else:
        #    # m = (sensitivity + specificity)/2 # balanced accuracy
        #    raise NotImplementedError("Balanced accuracy metric is not implemented")

        return (pred_true_cnt - true_cnt) / true_cnt  # (batchsize 1024)

    # @staticmethod
    # def bin_prd_clsf_info(y_true, y_pred, threshold=0.5, N_MORE=True, epsilon=1e-12):
    #    """
    #    refer to this: https://stats.stackexchange.com/questions/49579/balanced-accuracy-vs-f-1-score

    #    Both F1 and b_acc are metrics for classifier evaluation, that (to some extent) handle class imbalance. Depending
    #     of which of the two classes (N or P) outnumbers the other, each metric is outperforms the other.

    #    1) If N >> P, f1 is a better.

    #    2) If P >> N, b_acc is better.

    #    For code: refer to this: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/70841

    #    :param y_true:
    #    :param y_pred:
    #    :param threshold:
    #    :return: accuracy, f1 for this batch... not the global one, we need to be careful!!
    #    """
    #    if FOCAL_LOSS_GAMMA == 2.0:
    #        threshold = 0.57
    #    elif FOCAL_LOSS_GAMMA == 1.0:
    #        threshold = (0.53+(0.722-0.097))/2  #(by...reading the test result..., found it changes every training... so useless)
    #    threshold = math_ops.cast(threshold, y_pred.dtype)
    #    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    #    y_true = math_ops.cast(y_true > threshold, y_pred.dtype)

    #    ground_pos = math_ops.reduce_sum(y_true) + epsilon
    #    correct_pos = math_ops.reduce_sum(math_ops.multiply(y_true, y_pred)) + epsilon
    #    predict_pos = math_ops.reduce_sum(y_pred) + epsilon

    #    precision = math_ops.div(correct_pos, predict_pos)
    #    recall = math_ops.div(correct_pos, ground_pos)

    #    if N_MORE:
    #        m = (2*recall*precision) / (precision+recall)
    #    else:
    #        #m = (sensitivity + specificity)/2 # balanced accuracy
    #        raise NotImplementedError("Balanced accuracy metric is not implemented")

    #    return m

    # so nine model for them...
    def build_identity_model(self, identity_out_num):
        # d.IDENTITY_COLUMNS  # with more than 500
        """build lstm model

        :param embedding_matrix:
        :param num_aux_targets:
        :return: the built model lstm
        """
        words = Input(shape=(None,))
        x = Embedding(
            *self.embedding_matrix.shape,
            weights=[self.embedding_matrix],
            trainable=False,
        )(words)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(CuDNNLSTM(int(LSTM_UNITS // 2), return_sequences=True))(x)

        hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x),])
        hidden = add(
            [hidden, Dense(int(DENSE_HIDDEN_UNITS // 2), activation="relu")(hidden)]
        )
        result = Dense(identity_out_num, activation="sigmoid")(hidden)
        # result_with_aux = Dense(num_aux_targets+1, activation='sigmoid')(hidden)

        model = Model(inputs=words, outputs=result)
        # model = Model(inputs=words, outputs=result_with_aux)
        # model.compile(loss='binary_crossentropy', optimizer='adam')
        try:
            # need to improve#for target it should be fine, they are positive related
            model.compile(
                loss="binary_crossentropy",
                optimizer="adam",
                metrics=[mean_absolute_error, binary_sensitivity, binary_specificity],
            )
        except Exception as e:
            logger.warning(e)
            # need to improve#for target it should be fine, they are positive related
            model.compile(
                loss="binary_crossentropy",
                optimizer="adam",
                metrics=[mean_absolute_error],
            )

        return model

    def run_identity_model(self, subgroup, features, labels, params):
        set_trace()

        prefix = params["prefix"]
        file_name = f"{prefix}_{subgroup}_0.hdf5"
        params["check_point_path"] = file_name
        pred_only = params.get("predict_only", False)
        if not pred_only:
            target_subgroup = labels[:, d.IDENTITY_COLUMNS.index(subgroup)]

        if RESTART_TRAIN_ID or not os.path.isfile(file_name):
            if pred_only:
                raise RuntimeError("Need to have a trained model to predict")
            model = self.build_identity_model(1)  # one model per identity...
            self.run_model_train(model, features, target_subgroup, params)
        else:
            logger.info("restore from the model file")
            model = load_model(
                file_name,
                custom_objects={
                    "AttentionRaffel": AttentionRaffel,
                    "binary_sensitivity": binary_sensitivity,
                    "binary_specificity": binary_specificity,
                },
            )

            logger.info(f"restore from the model file {file_name} -> done\n\n\n\n")

            continue_train = params.get("continue_train", False)
            if continue_train:
                params["starter_lr"] = params["starter_lr"] / 16
                logger.debug(f"continue train with learning rate /16")
                self.run_model_train(model, features, target_subgroup, params)

        if not pred_only:
            identity_predict = self.run_model(model, "predict", self.train_X)
            identity_predict_for_metric = identity_predict[self.identity_idx]

            s = binary_sensitivity_np(
                identity_predict_for_metric.reshape(-1), target_subgroup
            )
            logger.info(f"for {subgroup}, predict_sensitivity is {s}")

            predict_file_name = f"{prefix}_{subgroup}_pred.pkl"
            id_preds = {}
            id_preds[subgroup] = identity_predict
            pickle.dump(identity_predict, open(predict_file_name, "wb"))
        else:  # only predict for test
            predict = self.run_model(model, "predict", features)
            logger.debug(
                f"in test set, predicted {subgroup}: mean {predict.mean()}, cnt {(predict>=0.5).sum()}, all_cnt {len(predict)}"
            )
            # self._val_index = val_ind  # for debug
            # pred = model.predict(self.train_X[val_ind], verbose=2)
            # self.oof_preds[val_ind] += pred
        # pickle.dump(self.oof_preds, open(prefix + "predicts", 'wb'))

    def build_lstm_model_customed(
        self,
        num_aux_targets,
        embedding_matrix,
        embedding_layer=None,
        with_aux=False,
        loss="binary_crossentropy",
        metrics=None,
        hidden_act="relu",
        with_BN=False,
    ):
        """build lstm model, non-binarized

        cls_reg: 0 for classification, 1 for regression(linear)
        metrics: should be a list
        with_aux: default False, don't know how to interpret...
        with_BN: ... BatchNormalization not work, because different subgroup different info, batch normalize will make it worse?
        """
        logger.debug(
            f"model detail: loss {loss}, hidden_act {hidden_act}, with_BN {with_BN}"
        )
        if num_aux_targets > 0 and not with_aux:
            raise RuntimeError("aux features numbers given but aux not enabled")
        if num_aux_targets <= 0 and with_aux:
            raise RuntimeError("aux features numbers invalid when aux enabled")

        words = Input(shape=(d.MAX_LEN,))  # (None, 180)
        if embedding_layer is not None:
            x = embedding_layer(words)
        else:
            x = Embedding(
                *embedding_matrix.shape, weights=[embedding_matrix], trainable=False
            )(words)

        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

        hidden = concatenate(
            [
                AttentionRaffel(d.MAX_LEN, name="attention_after_lstm")(x),
                GlobalMaxPooling1D()(x),  # with this 0.9125 ...(not enough test...)
                # GlobalAveragePooling1D()(x),  # a little worse to use this, 0.9124
            ]
        )

        activate_type = hidden_act
        if activate_type == "prelu":  # found it not working
            hidden = add(
                [hidden, PReLU()(Dense(DENSE_HIDDEN_UNITS, activation=None)(hidden))]
            )
            if with_BN:
                hidden = BatchNormalization()(hidden)
            hidden = add(
                [hidden, PReLU()(Dense(DENSE_HIDDEN_UNITS, activation=None)(hidden))]
            )
        else:
            hidden = add(
                [hidden, Dense(DENSE_HIDDEN_UNITS, activation=activate_type)(hidden)]
            )
            if with_BN:
                hidden = BatchNormalization()(hidden)
            hidden = add(
                [hidden, Dense(DENSE_HIDDEN_UNITS, activation=activate_type)(hidden)]
            )

        logit = Dense(1, activation=None)(hidden)
        logit = Lambda(
            lambda pre_sigmoid: FOCAL_LOSS_GAMMA_NEG_POS * pre_sigmoid
            + FOCAL_LOSS_BETA_NEG_POS
        )(logit)
        result = Activation("sigmoid")(logit)

        if with_aux:
            aux_result = Dense(num_aux_targets, activation="sigmoid")(hidden)
            model = Model(inputs=words, outputs=[result, aux_result])
            model.compile(
                loss=[loss, "binary_crossentropy"],
                optimizer="adam",
                loss_weights=[1.0, 0.25],
                metrics=metrics,
            )
        else:
            model = Model(inputs=words, outputs=result)
            model.compile(loss=loss, optimizer="adam", metrics=metrics)

        return model

    def build_lstm_model(self, num_aux_targets):
        words = Input(shape=(None,))
        x = Embedding(
            *self.embedding_matrix.shape,
            weights=[self.embedding_matrix],
            trainable=False,
        )(words)
        logger.info(
            "Embedding fine, here the type of embedding matrix is {}".format(
                type(self.embedding_matrix)
            )
        )

        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

        hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x),])
        hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation="relu")(hidden)])
        hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation="relu")(hidden)])
        result = Dense(1, activation="sigmoid")(hidden)
        aux_result = Dense(num_aux_targets, activation="sigmoid")(hidden)

        model = Model(inputs=words, outputs=[result, aux_result])
        # model = Model(inputs=words, outputs=result_with_aux)
        # model.compile(loss='binary_crossentropy', optimizer='adam')

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(0.005),
            metrics=[KaggleKernel.bin_prd_clsf_info],
        )
        # for binary_crossentropy, the implementation is in  tensorflow/tensorflow/python/keras/backend.py
        #       bce = target * math_ops.log(output + epsilon())
        #       bce += (1 - target) * math_ops.log(1 - output + epsilon())
        # return -bce # binary cross entropy

        # for binary accuraty:
        # def binary_accuracy(y_true, y_pred, threshold=0.5):
        #   threshold = math_ops.cast(threshold, y_pred.dtype)
        #   y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
        #   return K.mean(math_ops.equal(y_true, y_pred), axis=-1)

        # fit a line for linear regression, we use least square error, (residuals), norm, MLE
        # logistic regression uses the log(odds) on the y-axis, as log(odds) push points to
        # + , - infinity, we cannot use least square error, we use maximum likelihood, the line
        # can still be imagined to there. for every w guess, we have a log-likelihood for the
        # line. we need to find the ML line
        return model

    def run_lstm_model(
        self,
        final_train=False,
        n_splits=NUM_MODELS,
        predict_ones_with_identity=True,
        params=None,
        val_mask=None,
    ):
        # TODO condiser use train test split as
        """
        train_df, validate_df = model_selection.train_test_split(train, test_size=0.2)
        logger.info('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))"""

        # self.oof_preds = np.zeros((self.train_X.shape[0], 1 + self.train_y_aux.shape[1]))
        test_preds = np.zeros((self.to_predict_X_all.shape[0]))
        prefix = params["prefix"]
        re_train = params["re-start-train"]
        predict_only = params["predict-only"]
        logger.debug(f"pred only? {predict_only}\n\n\n")
        # sample_weights = params.get('sample_weights')
        train_data = params.get("train_data", None)
        train_y_aux_passed = params.get("train_y_aux")
        val_y_aux_passed = params.get("val_y_aux")
        patience = params.get("patience", 3)
        bin_target = params.get("binary")
        f_g = params.get("fortify_subgroup")

        if train_data is None:
            train_X = self.train_X
            train_y = self.train_y
            train_y_aux = self.train_y_aux
        else:
            train_X, train_y = train_data
            train_y_aux = train_y_aux_passed
            val_y_aux = val_y_aux_passed
        if train_y_aux is None and not NO_AUX:
            raise RuntimeError("Need aux labels to train")
        if val_y_aux is None and not NO_AUX:
            raise RuntimeError("Need aux labels to validate")

        val_data = params.get("val_data")
        if val_data is None:  # used to dev evaluation
            val_X = self.train_X_identity
        else:
            val_X, val_y = val_data

        prefix += "G{:.1f}".format(FOCAL_LOSS_GAMMA)

        if self.embedding_layer is None:
            self.embedding_layer = Embedding(
                *self.embedding_matrix.shape,
                weights=[self.embedding_matrix],
                trainable=False,
            )

            del self.embedding_matrix
            gc.collect()

        # build one time, then reset if needed
        if NO_AUX:
            # we could load with file name, then remove and save to new one
            h5_file = prefix + "_attention_lstm_NOAUX_" + f".hdf5"
        else:
            # we could load with file name, then remove and save to new one
            h5_file = prefix + "_attention_lstm_" + f".hdf5"

        logger.debug(h5_file)

        starter_lr = params.get("starter_lr", STARTER_LEARNING_RATE)
        # model thing
        load_from_model = False
        if re_train or not os.path.isfile(h5_file):
            logger.debug(
                f"re_train is {re_train}, file {h5_file} exists? {os.path.isfile(h5_file)}"
            )

            if NO_AUX:
                if FOCAL_LOSS:
                    model = self.build_lstm_model_customed(
                        0,
                        None,
                        self.embedding_layer,
                        with_aux=False,
                        loss=binary_crossentropy_with_focal,
                        metrics=[binary_crossentropy, mean_absolute_error,],
                    )
                else:
                    model = self.build_lstm_model_customed(
                        0,
                        None,
                        self.embedding_layer,
                        with_aux=False,
                        metrics=[mean_absolute_error,],
                    )
            else:
                logger.debug("using loss=binary_crossentropy_with_focal_seasoned")
                model = self.build_lstm_model_customed(
                    len(self.train_y_aux[0]),
                    None,
                    self.embedding_layer,
                    with_aux=True,
                    loss=binary_crossentropy_with_focal,
                    metrics=[binary_crossentropy, mean_absolute_error],
                )
            self.model = model
            model.summary()
            logger.info("build model -> done\n\n\n")

        else:
            model = load_model(
                h5_file,
                custom_objects={
                    "binary_crossentropy_with_focal_seasoned": binary_crossentropy_with_focal_seasoned,
                    "binary_crossentropy_with_focal": binary_crossentropy_with_focal,
                    "AttentionRaffel": AttentionRaffel,
                },
            )
            load_from_model = True
            starter_lr = starter_lr * LEARNING_RATE_DECAY_PER_EPOCH ** (EPOCHS)
            self.model = model
            logger.debug("restore from the model file {} -> done\n\n\n".format(h5_file))

        if FINAL_DEBUG and load_from_model:
            pred = model.predict(val_X, verbose=2, batch_size=BATCH_SIZE)
            self.check_preds_in_val(pred, val_mask, run_times=0)

        run_times = 0
        better_run_times = 0
        final_score = 0
        for fold in range(n_splits):
            # K.clear_session()  # so it will start over
            if fold > 0:
                reinitLayers(model)
                starter_lr = starter_lr / 8  # as lstm won't be re-initialized

            ckpt = ModelCheckpoint(h5_file, save_best_only=True, verbose=1)
            early_stop = EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=1,
                patience=patience,
                restore_best_weights=True,
            )

            if not predict_only:
                prog_bar_logger = NBatchProgBarLogger(
                    display_per_batches=int(1600000 / 20 / BATCH_SIZE),
                    early_stop=True,
                    patience_displays=3,
                )

                if (
                    f_g is not None and bin_target
                ):  # only binary for subgroup, just try to see if it helps
                    # we also need to filter out the subgroup data (in outer layer now)
                    val_y = val_y >= 0.5

                train_labels = train_y if NO_AUX else [train_y, train_y_aux]
                val_labels = val_y if NO_AUX else [val_y, val_y_aux]

                model.fit(
                    train_X,
                    train_labels,
                    # validation_split=val_split,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    # sample_weight=sample_weights,
                    # steps_per_epoch=int(len(self.train_X)*False95/BATCH_SIZE),
                    verbose=0,  # prog_bar_logger will handle log, so set to 0 here
                    # validation_data=(self.train_X[val_ind], self.train_y[val_ind]>0.5),
                    validation_data=(val_X, val_labels),
                    callbacks=[
                        LearningRateScheduler(
                            # STARTER_LEARNING_RATE = 1e-2
                            # LEARNING_RATE_DECAY_PER_EPOCH = 0.7
                            lambda e: starter_lr * (LEARNING_RATE_DECAY_PER_EPOCH ** e),
                            verbose=1,
                        ),
                        early_stop,
                        ckpt,
                        prog_bar_logger,
                    ],
                )

            pred = model.predict(val_X, verbose=2, batch_size=BATCH_SIZE)
            final_score_this_run = self.check_preds_in_val(
                pred, val_mask, run_times=run_times
            )
            improve_flag = False
            if final_score_this_run > final_score:
                final_score = final_score_this_run
                improve_flag = True
            else:
                improve_flag = False

            if not NOT_PRD:
                # if final_train:
                test_result = model.predict(self.to_predict_X_all, verbose=2)
                run_times += 1
                if NO_AUX:
                    test_result_column = np.array(test_result).ravel()
                else:
                    # the shape of preds, is [0] is the predict,[1] for aux
                    test_result_column = np.array(test_result[0]).ravel()
                self.save_result(
                    test_result_column, filename=f"submission_split_{run_times}.csv"
                )

                if improve_flag:
                    logger.debug("better this time, will update submission.csv\n\n\n")
                    better_run_times += 1
                    test_preds += test_result_column
                    test_preds_avg = test_preds / better_run_times
                    # save everything in case failed in some folds
                    self.save_result(test_preds_avg)

    def check_preds_in_val(self, pred, val_mask, run_times=0):
        if DEBUG_TRACE:
            set_trace()
        if not NO_AUX:
            preds = np.array(pred[0]).ravel()
        else:
            preds = np.array(pred).ravel()

        if DEBUG_TRACE:
            set_trace()
        # preds = 1 / (1+np.exp(-preds))

        self.train_df.loc[val_mask, "lstm"] = preds
        if True:
            self.train_df[d.VAL_ERR_COLUMN] = 0
            # self.train_df.loc[val_mask, 'lstm'] = preds
            self.train_df.loc[val_mask, d.VAL_ERR_COLUMN] = (
                preds - self.train_df.loc[val_mask, d.TARGET_COLUMN]
            )
            dstr = self.target_analyzer.get_err_distribution(self.train_df, val_mask)
            for k, v in dstr.items():
                logger.debug(f"error info: {k}, {v[1]}")
        return self.calculate_metrics_and_print(
            filename_for_print=f"metrics_log_{run_times}.txt",
            validate_df_with_preds=kernel.train_df[val_mask],
            benchmark_base=kernel.train_df[val_mask],
        )

    #        test_preds /= run_times
    #        self.save_result(test_preds)

    def save_result(self, predictions, filename=None):
        if self.emb.test_df_id is None:
            self.emb.read_train_test_df()
        submission = pd.DataFrame.from_dict(
            {
                "id": self.emb.test_df_id,
                # 'id': test_df.id,
                "prediction": predictions,
            }
        )
        if filename is not None:
            submission.to_csv(filename, index=False)
        else:
            submission.to_csv("submission.csv", index=False)

    def prepare_second_stage_data_index(
        self, y_pred, subgroup="white", only_false_postitive=False, n_splits=5
    ):
        """

        :param y_pred: # might needed, compare pred and target, weight the examples
        :param subgroup:
        :param only_false_postitive: only train the ones with higher then actual ones?
        :return: X, y pair
        """
        df = self.id_validate_df
        subgroup_df = df[df[subgroup]]
        subgroup_idx = subgroup_df.index
        # size = len(subgroup_idx)

        splits = list(
            KFold(n_splits=n_splits, random_state=2019, shuffle=True).split(
                subgroup_idx
            )
        )  # just use sklearn split to get id and it is fine. For text thing,
        # just do 1 fold, later we can add them all back
        tr_ind, val_ind = splits[0]
        return subgroup_df.iloc[tr_ind].index, subgroup_df.iloc[val_ind].index

    def build_res_model(
        self,
        subgroup,
        loss="binary_crossentropy",
        metrics=None,
        hidden_act="relu",
        with_BN=False,
    ):
        if self.model is None:
            logger.debug("Start loading model")
            h5_file = "/proc/driver/nvidia/G2.0_attention_lstm_NOAUX_0.hdf5"
            self.model = load_model(
                h5_file,
                custom_objects={
                    "binary_crossentropy_with_focal": binary_crossentropy_with_focal,
                    "AttentionRaffel": AttentionRaffel,
                },
            )
            logger.debug("Done loading model")
        base_model = self.model

        # add_1 = base_model.get_layer('add_1')
        add_2 = base_model.get_layer("add_2")
        main_logit = base_model.get_layer("dense_3")

        #       after add_1, add another dense layer, (so totally not change exited structure(might affect other subgroup?)
        #       then add this with add_2, finally go to sigmoid function (loss function no change ....) (should use small learning rate)
        # hidden = concatenate([
        #    add_2.output,
        #    Dense(RES_DENSE_HIDDEN_UNITS, activation=hidden_act, name="res_features_recombination")(add_1.output)
        # ], name='cat_'+subgroup+'_res_to_main')
        # result = Dense(1, activation='sigmoid', name='res_main_together')(hidden) -> not good
        res_recombination = Dense(
            RES_DENSE_HIDDEN_UNITS,
            activation=hidden_act,
            name="res_features_recombination",
        )(add_2.output)
        hidden = add(
            [
                res_recombination,
                Dense(RES_DENSE_HIDDEN_UNITS, activation=hidden_act, name="res_res")(
                    res_recombination
                ),
            ],
            name="res_res_added",
        )
        res_logit = Dense(1, activation=None, name="res_logit")(hidden)
        logit = add([res_logit, main_logit.output], name="whole_logit")
        result = Activation("sigmoid", name="whole_predict")(logit)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
        # compile the model (should be done *after* setting layers to non-trainable)
        model = Model(inputs=base_model.input, outputs=result)

        model.compile(optimizer="adam", loss=loss, metrics=metrics)
        return model

    def run_model_train(
        self, model, X, y, params, use_split=False, n_splits=5, y_aux=None
    ):
        if use_split:
            assert n_splits > 1
            # just use sklearn split to get id and it is fine. For text thing,
            splits = list(
                KFold(n_splits=n_splits, random_state=2019, shuffle=True).split(X)
            )
        else:
            n_splits = 1  # for the following for loop

        prefix = params["prefix"]
        starter_lr = params["starter_lr"]
        validation_split = params.get("validation_split", 0.05)
        epochs = params.get("epochs", EPOCHS)
        lr_decay = params.get("lr_decay", LEARNING_RATE_DECAY_PER_EPOCH)
        patience = params.get("patience", 3)
        display_per_epoch = params.get("display_per_epoch", 5)
        display_verbose = params.get("verbose", 2)
        no_check_point = params.get("no_check_point", False)
        passed_check_point_file_path = params.get("check_point_path", None)

        prefix += "G{:.1f}".format(FOCAL_LOSS_GAMMA)

        for fold in range(n_splits):  # will need to do this later
            # K.clear_session()  # so it will start over todo fix K fold
            if use_split:
                tr_ind, val_ind = splits[fold]
                logger.info(
                    "%d train comments, %d validate comments" % (tr_ind, val_ind)
                )

            else:
                tr_ind, val_ind = [True] * len(X), [False] * len(X)

            if NO_AUX:
                # we could load with file name, then remove and save to new one
                h5_file = prefix + "_attention_lstm_NOAUX_" + f"{fold}.hdf5"
            else:
                # we could load with file name, then remove and save to new one
                h5_file = prefix + "_attention_lstm_" + f"{fold}.hdf5"

            if passed_check_point_file_path is not None:
                h5_file = passed_check_point_file_path

            logger.debug(f"using checkpoint files: {h5_file}")

            early_stop = EarlyStopping(
                monitor="val_loss", mode="min", verbose=1, patience=patience
            )

            # data thing
            if NO_AUX:
                y_train = y[tr_ind]
                y_val = y[val_ind]
            else:
                y_train = [y[tr_ind], y_aux[tr_ind]]
                y_val = [y[val_ind], y_aux[val_ind]]

            callbacks = [
                LearningRateScheduler(
                    lambda e: starter_lr * (lr_decay ** e), verbose=1
                ),
                early_stop,
            ]

            if not no_check_point:
                ckpt = ModelCheckpoint(h5_file, save_best_only=True, verbose=1)
                callbacks.append(ckpt)

            if display_verbose == 1:
                verbose = 0
                prog_bar_logger = NBatchProgBarLogger(
                    display_per_batches=int(
                        len(tr_ind) / display_per_epoch / BATCH_SIZE
                    ),
                    early_stop=True,
                    patience_displays=patience,
                )
                callbacks.append(prog_bar_logger)
            else:  # 0 or 2
                verbose = display_verbose

            logger.debug(
                f"{len(tr_ind)} training, {validation_split*len(tr_ind)} validation in fit"
            )

            model.fit(
                X[tr_ind],
                y[tr_ind],
                validation_split=validation_split,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
            )

    def run_model(self, model, mode, X, y=None, params={}, n_splits=0):
        # checkpoint_predictions = []
        # weights = []
        if mode == "train":
            self.run_model_train(model, X, y, params, n_splits > 1, n_splits)
        elif mode == "predict":
            pred = model.predict(X, verbose=2, batch_size=BATCH_SIZE)
            return pred

        # if predict_ones_with_identity:
        #    return model.predict(self.train_X_identity, verbose=2, batch_size=BATCH_SIZE)

    def locate_data_in_np_train(self, index):
        """

        :param index: must be for the index of range 405130
        :return:
        """
        return self.train_X[index], self.train_y[index], self.train_y_aux[index]

    def locate_subgroup_index_in_np_train(self, subgroup):
        df = self.id_validate_df
        index = df[df[subgroup]].index
        return index

    def locate_subgroup_data_in_np_train(self, subgroup):
        """

        :param index: must be for the index of range 405130
        :return:
        """
        index = self.locate_subgroup_index_in_np_train(subgroup)
        return self.locate_data_in_np_train(index)

    def to_identity_index(self, index):
        """

        :param index: from 1.8m range index
        :return: to 0.4 m range index in identity data
        """
        df = self.id_validate_df

        # selected the items
        return [df.index.get_loc(label) for label in index]

    def _get_identities(self):
        """
        No need to use this function, all identities are marked

        :return:
        """
        prefix = self.emb.BIN_FOLDER
        # if os.path.isfile(prefix+'train_df.pd'):
        if False:
            self.train_df = pickle.load(open(prefix + "train_df.pd", "rb"))
        else:
            for g in d.IDENTITY_COLUMNS:
                pred = pickle.load(open(f"{prefix}_{g}_pred.pkl", "rb"))
                self.train_df[f"{g}_pred"] = pred

        for g in d.IDENTITY_COLUMNS:
            self.train_df.loc[self.identity_idx, f"{g}_pred"] = self.train_df.loc[
                self.identity_idx, g
            ]

    def get_identities_for_training(self):
        if not self.id_used_in_train:
            #            if not FINAL_SUBMIT:
            # in test set, around 10% data will be with identities (lower than training set)
            logger.debug("Use 90% identity data")
            id_df = self.train_df.loc[self.identity_idx]
            # 40,000 remained for val
            id_train_df = id_df.sample(frac=0.9, random_state=2019)
            id_train_df_idx = id_train_df.index
            #            else:  # we still need these ... for the early stop thing!!!
            #                logger.debug("Use 100% identity data")  # in test set, around 10% data will be with identities (lower than training set)
            #                id_train_df_idx = self.identity_idx

            self.train_mask[id_train_df_idx] = 1

            self.id_used_in_train = True
            for g in d.IDENTITY_COLUMNS:
                # column to keep recored what data is used in training, used in data_prepare module...
                self.train_df[g + "_in_train"] = 0.0
                # only the ones larger than 0.5 will ? how about negative example?
                self.train_df[g + "_in_train"].loc[id_train_df_idx] = self.train_df[
                    g
                ].loc[id_train_df_idx]

    def prepare_weight_for_subgroup_balance(self):
        """ to see how other people handle weights [this kernel](https://www.kaggle.com/thousandvoices/simple-lstm)
            sample_weights = np.ones(len(x_train), dtype=np.float32)
            # more weights for the ones with identities, more identities, more weights
            sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
            # the toxic ones, reverse identity (without identity)(average 4~8), so more weights on toxic one without identity
            sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
            # none toxic, non-toxic, with identity, more weight for this, so the weights are more or less balanced
            sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
            sample_weights /= sample_weights.mean()

        And we know the identies now, so we balance all the ones,
        for every subgroup, we calculate the related weight to balance
        """
        self.train_df = self.emb.train_df
        # self._get_identities()  # for known ones, just skip
        analyzer = d.TargetDistAnalyzer(self.train_df)
        self.target_analyzer = analyzer
        o = analyzer.get_distribution_overall()
        self.get_identities_for_training()
        # for subgroup, use 0.5 as the limit, continuous info not used... anyway, we try first
        gs = analyzer.get_distribution_subgroups()

        # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
        balance_scheme_subgroups = BALANCE_SCHEME_SUBGROUPS
        # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
        balance_scheme_across_subgroups = BALANCE_SCHEME_ACROSS_SUBGROUPS
        # balance_scheme_target_splits = 'target_bucket_same_for_target_splits' # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
        # not work, because manual change will corrupt orignial information?
        balance_scheme_target_splits = BALANCE_SCHEME_TARGET_SPLITS
        balance_AUC = BALANCE_SCHEME_AUC

        # need a parameter for all pos v.s. neg., and for different
        def add_weight(balance_scheme_subgroups, balance_group=False):
            # target value, how do we balance?
            # (First we equalize them, then re-balance), just try different balance
            ones_weights = np.ones(len(self.train_df), dtype=np.float32)
            # sample_weights = ones_weights.copy()

            gs_weights_ratio = {}
            gs_weights = {}
            background_target_ratios = np.array([dstr[2] for dstr in o])

            if balance_scheme_subgroups == "target_bucket_same_for_subgroups":
                # compare with the background one, then change the weights to the same scale
                for g, v in gs.items():
                    gs_weights[g] = ones_weights.copy()  # initial, ones
                    # v is the distribution for ONE subgroup for 0~1 11 target types
                    gs_weights_ratio[g] = np.divide(
                        background_target_ratios, np.array([dstr[2] for dstr in v])
                    )
                    for target_split_idx, ratio in enumerate(gs_weights_ratio[g]):
                        # [3] is the index
                        split_idx_in_df = v[target_split_idx][3]
                        gs_weights[g][split_idx_in_df] *= ratio

            # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
            if balance_scheme_across_subgroups == "more_for_low_score":
                subgroup_weights = {}
                subgroup_weights["homosexual_gay_or_lesbian"] = 4
                subgroup_weights["black"] = 3
                subgroup_weights["white"] = 3
                subgroup_weights["muslim"] = 2.5
                subgroup_weights["jewish"] = 4
                """
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
                """
                for g in subgroup_weights.keys():
                    subgroup_dist = gs[g]
                    for dstr in subgroup_dist:
                        split_idx_in_df = dstr[3]
                        # the ones with identities will be added
                        gs_weights[g][split_idx_in_df] *= subgroup_weights[g]

            # shape will be [sample nubmers , subgroups] as some sample might be in two groups
            weights_changer = np.transpose([v for v in gs_weights.values()])
            weights_changer_max = np.amax(weights_changer, axis=1)
            weights_changer_min = np.amin(weights_changer, axis=1)
            weights_changer_mean = np.mean(weights_changer, axis=1)
            weights_changer_merged = ones_weights.copy()
            weights_changer_merged[weights_changer_mean > 1] = weights_changer_max[
                weights_changer_mean > 1
            ]
            weights_changer_merged[weights_changer_mean < 1] = weights_changer_min[
                weights_changer_mean < 1
            ]

            sample_weights = weights_changer_merged

            if balance_AUC == "more_bp_sn":
                # self.train_df, contains all info
                benchmark_base = (
                    self.train_df[d.IDENTITY_COLUMNS + [d.TARGET_COLUMN, d.TEXT_COLUMN]]
                    .fillna(0)
                    .astype(np.bool)
                )
                # the idx happen to be the iloc value
                judge = d.BiasBenchmark(benchmark_base, threshold=0.5)
                # converted to binary in judge initailization function
                id_validate_df = judge.validate_df
                toxic_bool_col = id_validate_df[d.TARGET_COLUMN]
                contain_identity_bool_col = id_validate_df[d.IDENTITY_COLUMNS].any(
                    axis=1
                )

                weights_auc_balancer = ones_weights.copy() / 4
                # for subgroup postitive, will be 0.5 weight
                weights_auc_balancer[contain_identity_bool_col] += 1 / 4
                # BPSN, BP part (0.5 weights)
                weights_auc_balancer[toxic_bool_col & ~contain_identity_bool_col] += (
                    1 / 4
                )
                # still BPSN, SN part (0.75 weights)
                weights_auc_balancer[~toxic_bool_col & contain_identity_bool_col] += (
                    1 / 4
                )

                sample_weights = np.multiply(sample_weights, weights_auc_balancer)

            wanted_split_ratios = None

            if balance_scheme_target_splits == "target_bucket_same_for_target_splits":
                wanted_split_ratios = [1 / len(background_target_ratios)] * len(
                    background_target_ratios
                )
            elif balance_scheme_target_splits == "target_bucket_extreme_positive":
                # 0 0.1 0.2 0.3 ... 1  # good
                wanted_split_ratios = [2, 2, 2, 2, 2, 10, 15, 20, 20, 15, 10]

            if wanted_split_ratios is not None:
                assert len(wanted_split_ratios) == len(background_target_ratios)
                for target_split_idx, ratio in enumerate(background_target_ratios):
                    idx_for_split = o[target_split_idx][3]
                    # 1/len(b_t_r) is what we want
                    sample_weights[idx_for_split] *= (
                        wanted_split_ratios[target_split_idx] / ratio
                    )

            sample_weights /= sample_weights.mean()  # normalize
            return sample_weights

        weights = add_weight(balance_scheme_subgroups)

        return weights

    def prepare_train_labels(
        self,
        train_y_all,
        train_mask,
        custom_weights=False,
        with_aux=False,
        train_y_aux=None,
        sample_weights=None,
        fortify_subgroup=None,
    ):
        val_mask = np.invert(kernel.train_mask)  # this is whole train_mask

        if fortify_subgroup is not None:
            # only use the subgroup data
            train_mask = train_mask & (
                self.train_df[fortify_subgroup + "_in_train"] >= 0.5
            )

        train_X = kernel.train_X_all[train_mask]
        val_X = kernel.train_X_all[val_mask]

        if not custom_weights:
            if with_aux:
                return (
                    train_X,
                    val_X,
                    train_y_all[train_mask],
                    train_y_aux[train_mask],
                    train_y_all[val_mask],
                    train_y_aux[val_mask],
                )
            else:
                return train_X, val_X, train_y_all[train_mask], train_y_all[val_mask]
        else:
            # credit to https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution
            if sample_weights is None:
                raise RuntimeError(
                    "sample weights cannot be None if use custom_weights"
                )
            assert len(train_y_all) == len(sample_weights)
            if with_aux:
                return (
                    train_X,
                    val_X,
                    np.vstack([train_y_all, sample_weights]).T[train_mask],
                    train_y_aux[train_mask],
                    train_y_all[val_mask],
                    train_y_aux[val_mask],
                )
            else:
                return (
                    train_X,
                    val_X,
                    np.vstack([train_y_all, sample_weights]).T[train_mask],
                    train_y_all[val_mask],
                )

    def res_subgroup(self, subgroup, y_pred):
        # first prepare the data
        # 1. the subgroup
        # 2. only the mis classified ones (will overfit to this?)  # it just like a res net... so just add a res net...
        # idx_train, idx_val must be complementary for subgroup
        idx_train, idx_val = self.prepare_second_stage_data_index(y_pred, subgroup)
        # prepare the network (load from the existed model, only change the last perception layer(as data not enough), then just train with the error related data)
        # 1. load the model
        X, y, _ = self.locate_data_in_np_train(idx_train)
        graph1 = Graph()
        with graph1.as_default():
            session1 = Session()
            with session1.as_default():
                h5_file = "/proc/driver/nvidia/white_G2.0_attention_lstm_NOAUX_0.hdf5"
                if RESTART_TRAIN_RES or not os.path.isfile(h5_file):
                    self.res_model = self.build_res_model(
                        subgroup,
                        loss=binary_crossentropy_with_focal,
                        metrics=[binary_crossentropy, mean_absolute_error,],
                    )
                    self.res_model.summary()

                    self.run_model(
                        self.res_model,
                        "train",
                        X,
                        y,
                        params={
                            "prefix": "/proc/driver/nvidia/" + subgroup + "_",
                            "starter_lr": STARTER_LEARNING_RATE / 8,
                            "epochs": 70,
                            "patience": 5,
                            "lr_decay": 0.8,
                            "validation_split": 0.05,
                            "no_check_point": True,
                        },
                    )
                else:  # file exit and restart_train==false
                    logger.debug(f"load res model from {h5_file}")
                    self.res_model = load_model(
                        h5_file,
                        custom_objects={
                            "binary_crossentropy_with_focal": binary_crossentropy_with_focal,
                            "AttentionRaffel": AttentionRaffel,
                        },
                    )
                    self.res_model.summary()

                # y_res_pred = self.run_model(self.res_model, 'predict', X[idx_val])  # X all data with identity
                subgroup_idx = self.locate_subgroup_index_in_np_train(subgroup)
                mapped_subgroup_idx = self.to_identity_index(subgroup_idx)

                X_all, y_all = self.train_X_identity, self.train_y_identity

                logger.info("Start predict for all identities")
                y_res_pred_all_group = self.run_model(self.res_model, "predict", X_all)
                logger.info("Done predict for all identities")

                y_res_pred_all_train_val = y_res_pred_all_group[mapped_subgroup_idx]

                y_res_pred = y_res_pred_all_group[
                    self.to_identity_index(idx_val)
                ]  # find the index

        # 2. modify, change to not training
        #       after add_1, add another dense layer, (so totally not change exited structure(might affect other subgroup?)
        #       then add this with add_2, finally go to sigmoid function (loss function no change ....) (should use small learning rate)
        # 3. train only the error ones
        # 4. predict (the whole out)

        # finally, predict, merge the data
        # logger.debug(f'Predict for val {subgroup} comments, {len(y_res_pred)} items')  # change too small, so ignore
        # self.res_combine_pred_print_result(subgroup, y_pred, y_res_pred, idx_train, idx_val)  # remove idx_train, add idx_val, then calculate auc

        # should use cross validation
        logger.debug(
            f"Predict for this {subgroup} comments, {len(y_res_pred_all_train_val)} items"
        )
        self.res_combine_pred_print_result(
            subgroup, y_pred, y_res_pred_all_train_val, [], subgroup_idx, detail=False
        )  # remove idx_train, add idx_val, then calculate auc

        # should use cross validation
        logger.debug(f"Predict for all {subgroup} comments, {len(y_res_pred)} items")
        self.calculate_metrics_and_print(
            y_res_pred_all_group, detail=False
        )  # only the ones with identity

    # remove idx_train, add idx_val, then calculate auc
    def res_combine_pred_print_result(
        self, subgroup, y_pred, y_res_pred, idx_train, idx_val, detail=False
    ):
        id_df = copy.deepcopy(self.judge.validate_df)
        assert len(idx_train) + len(idx_val) == len(id_df[id_df[subgroup]])

        assert id_df.shape[0] == len(y_pred)

        model_name = "res_" + subgroup
        # there are comments mention two identity, so our way might not be good
        id_df[model_name] = y_pred
        # not iloc, both are index from the 1.8 Million data
        id_df.loc[idx_val, id_df.columns.get_loc(model_name)] = y_res_pred

        logger.debug(
            f"Res update for {subgroup}, {len(idx_val)} items predicted by res model"
        )

        self.calculate_metrics_and_print(
            validate_df_with_preds=id_df,
            model_name=model_name,
            detail=detail,
            file_for_print="metrics_log.txt",
        )

    def run_bias_auc_model(self):
        """
        need to prepare data, then train network to handle the bias thing
        we use data (identity(given value), comment text) as feature, to recalculate target, and reduce bias

        after build right model, then use predicted features to do the same prediction

        :return:
        """
        pass

    def load_identity_data_idx(self):
        if self.identity_idx is None:
            # to train the identity
            (
                self.train_X_identity,
                self.train_y_identity,
                self.identity_idx,
            ) = self.emb.get_identity_train_data_df_idx()

    def calculate_metrics_and_print(
        self,
        filename_for_print="metrics_log.txt",
        preds=None,
        threshold=0.5,
        validate_df_with_preds=None,
        model_name="lstm",
        detail=True,
        benchmark_base=None,
    ):
        file_for_print = open(filename_for_print, "w")

        self.emb.read_train_test_df(train_only=True)
        self.load_identity_data_idx()
        if benchmark_base is None:
            benchmark_base = self.train_df.loc[self.identity_idx]

        # if self.judge is None:  # no .... different threshold need to recalculate in the new judge
        # the idx happen to be the iloc value
        self.judge = d.BiasBenchmark(benchmark_base, threshold=threshold)
        self.id_validate_df = self.judge.validate_df

        if model_name == d.MODEL_NAME:
            if preds is not None:
                logger.debug(f"{model_name} result for {len(preds)} items:")
            if validate_df_with_preds is not None:
                logger.debug(
                    f"{model_name} result for {len(validate_df_with_preds)} items:"
                )
            if validate_df_with_preds is not None:
                (
                    value,
                    score_comp,
                    bias_metrics,
                    subgroup_distribution,
                    overall_distribution,
                ) = self.judge.calculate_benchmark(
                    validate_df=validate_df_with_preds, model_name=model_name
                )
            else:
                (
                    value,
                    score_comp,
                    bias_metrics,
                    subgroup_distribution,
                    overall_distribution,
                ) = self.judge.calculate_benchmark(preds)
        elif model_name.startswith("res"):
            logger.debug(
                f"{model_name} result for {len(validate_df_with_preds)} items in background"
            )
            (
                value,
                score_comp,
                bias_metrics,
                subgroup_distribution,
                overall_distribution,
            ) = self.judge.calculate_benchmark(
                validate_df=validate_df_with_preds, model_name=model_name
            )

        bias_metrics_df = bias_metrics.set_index("subgroup")

        # only the ones with identity is predicted
        pickle.dump(bias_metrics_df, open("bias_metrics", "wb"))
        # only the ones with identity is predicted
        pickle.dump(subgroup_distribution, open("subgroup_dist", "wb"))

        # bias_metrics_df = pickle.load(open("bias_metrics", 'rb'))  # only the ones with identity is predicted
        # subgroup_distribution = pickle.load(open("subgroup_dist", 'rb'))  # only the ones with identity is predicted
        if not detail:
            return

        print(
            f"bias metrics details (AUC, BPSN, BNSP): {score_comp}", file=file_for_print
        )
        print(
            f"final metric: {value} for threshold {threshold} applied to '{d.TARGET_COLUMN}' column, ",
            file=file_for_print,
        )
        # print("\n{}".format(bias_metrics[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']]), file=file_for_print)
        print(
            "\n{}".format(bias_metrics[["subgroup", "subgroup_auc"]]),
            file=file_for_print,
        )

        print("### subgroup auc", file=file_for_print)

        def d_str(t):
            num, mean, std = t
            return f"{num}, {mean:.4}, {std:.4}"

        for d0 in subgroup_distribution:
            g = d0["subgroup"]
            m = "subgroup_auc"
            s = "subgroup_size"
            auc = "{:.4} {}".format(
                bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s]
            )
            print(
                "{:5.5} ".format(g)
                + auc
                + "\t"
                + d_str(d0[m][2])
                + "\t"
                + d_str(d0[m][3])
                + "\t"
                + d_str(d0[m][4])
                + "\t"
                + d_str(d0[m][5]),
                file=file_for_print,
            )

        print("### bpsn auc", file=file_for_print)
        for d0 in subgroup_distribution:
            g = d0["subgroup"]
            m = "bpsn_auc"
            s = "subgroup_size"
            auc = "{0:.4} {1}".format(
                bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s]
            )
            print(
                "{0:5.5} ".format(g)
                + auc
                + "\t"
                + d_str(d0[m][2])
                + "\t"
                + d_str(d0[m][3]),
                file=file_for_print,
            )

        print("### bnsp auc", file=file_for_print)
        for d0 in subgroup_distribution:
            g = d0["subgroup"]
            m = "bnsp_auc"
            s = "subgroup_size"
            auc = "{0:.4} {1}".format(
                bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s]
            )
            print(
                "{0:5.5} ".format(g)
                + auc
                + "\t"
                + d_str(d0[m][2])
                + "\t"
                + d_str(d0[m][3]),
                file=file_for_print,
            )

        print("### counts", file=file_for_print)
        # length thing
        for d0 in subgroup_distribution:
            g = d0["subgroup"]
            m = "subgroup_auc"
            s = "subgroup_size"
            auc = "{0:.4} {1}".format(
                bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s]
            )
            print(
                "{0:5.5} ".format(g)
                + auc
                + "\t"
                + str(d0[m][0])
                + "\t"
                + str(d0[m][1]),
                file=file_for_print,
            )

        print("### overall", file=file_for_print)
        g = "overall"
        m = d.OVERALL_AUC
        s = "subgroup_size"
        auc = "{0:.4} {1}".format(overall_distribution[m], overall_distribution[s])
        dist = overall_distribution["distribution"]
        print(
            f"{g:5.5} {auc}\tneg_tgt_pred_dis:{d_str(dist[2])}\tpos_tgt_pred_dis:{d_str(dist[3])}\noverall_pos_neg_cnt:\t{dist[0]}",
            file=file_for_print,
        )
        print(
            f"{g:5.5} {auc}\tneg_tgt_act_dis:{d_str(dist[4])}\tpos_tgt_act_dis:{d_str(dist[5])}",
            file=file_for_print,
        )

        file_for_print.close()
        file_for_print = open(filename_for_print, "r")
        print(file_for_print.read())

        return value  # final value


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument(
    "--train_steps", default=2, type=int, help="number of training steps"
)
parser.add_argument("--learning_rate", default=0.001, type=float, help="learing rate")


# FOCAL_LOSS_GAMMA = 2.
# ALPHA = 0.666
# FOCAL_LOSS_GAMMA = 0.
# ALPHA = 0.7(0.9121)  # 0.9(0.9122), 0.8(0.9123...) (no difference with no focal loss)
# GAMMA works better 2. with BS 1024
# GAMMA works better 1.5 with BS 512

# _debug_train_data = None

CONVERT_DATA = False
CONVERT_DATA_Y_NOT_BINARY = "convert_data_y_not_binary"
# given the pickle of numpy train data
CONVERT_TRAIN_DATA = "convert_train_data"
# given the pickle of numpy train data
CONVERT_ADDITIONAL_NONTOXIC_DATA = "CONVERT_ADDITIONAL_NONTOXIC_DATA"

EXCLUDE_IDENTITY_IN_TRAIN = True
TRAIN_DATA_EXCLUDE_IDENDITY_ONES = "TRAIN_DATA_EXCLUDE_IDENDITY_ONES"
DATA_ACTION_NO_NEED_LOAD_EMB_M = "DATA_ACTION_NO_NEED_LOAD_EMB_M"

NEG_RATIO = 1 - 0.05897253769515213


def binary_sensitivity_np(y_pred, y_true):
    threshold = 0.5
    # predict_false = y_pred <= threshold
    y_true = y_true > threshold
    predict_true = y_pred > threshold
    TP = np.multiply(y_true, predict_true)
    # FP = np.logical_and(y_true == 0, predict_true)

    # as Keras Tensors
    TP = TP.sum()
    # FP = FP.sum()

    sensitivity = TP / y_true.sum()
    return sensitivity


def binary_sensitivity(y_pred, y_true):
    """Compute the confusion matrix for a set of predictions.

    Parameters
    ----------
    y_pred   : predicted values for a batch if samples (must be binary: 0 or 1)
    y_true   : correct values for the set of samples used (must be binary: 0 or 1)

    Returns
    -------
    out : the specificity
    """
    threshold = 0.5
    TP = np.logical_and(K.eval(y_true) == 1, K.eval(y_pred) <= threshold)
    FP = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) > threshold)

    # as Keras Tensors
    TP = K.sum(K.variable(TP))
    FP = K.sum(K.variable(FP))

    sensitivity = TP / (TP + FP + K.epsilon())
    return sensitivity


def binary_specificity(y_pred, y_true):
    """Compute the confusion matrix for a set of predictions.

    Parameters
    ----------
    y_pred   : predicted values for a batch if samples (must be binary: 0 or 1)
    y_true   : correct values for the set of samples used (must be binary: 0 or 1)

    Returns
    -------
    out : the specificity
    """

    threshold = 0.5
    TN = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) <= threshold)
    FP = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) > threshold)

    # as Keras Tensors
    TN = K.sum(K.variable(TN))
    FP = K.sum(K.variable(FP))

    specificity = TN / (TN + FP + K.epsilon())
    return specificity


# search !!!!! tensorflow has auc.......................!!!!!!!!!!!!!!!!!!!!
def binary_auc_probability(y_true, y_pred, threshold=0.5, N_MORE=True, epsilon=1e-12):
    """
    refer to this: https://blog.revolutionanalytics.com/2016/11/calculating-auc.html

    The probabilistic interpretation is that if you randomly choose a positive case and a negative case, the probability that the positive case outranks the negative case according to the classifier is given by the AUC. This is evident from the figure, where the total area of the plot is normalized to one, the cells of the matrix enumerate all possible combinations of positive and negative cases, and the fraction under the curve comprises the cells where the positive case outranks the negative one.

    :param y_true:
    :param y_pred:
    :param threshold:
    :return: accuracy, f1 for this batch... not the global one, we need to be careful!!
    """

    # labels: y_true, scores: y_pred, N the size of sample
    # auc_probability < - function(labels, scores, N=1e7)
    # {
    #     pos < - sample(scores[labels], N, replace=TRUE)
    # neg < - sample(scores[!labels], N, replace = TRUE)
    # # sum( (1 + sign(pos - neg))/2)/N # does the same thing
    # (sum(pos > neg) + sum(pos == neg) / 2) / N  # give partial credit for ties
    # }

    # auc_probability(as.logical(category), prediction)

    threshold = math_ops.cast(threshold, y_pred.dtype)
    # y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_pred.dtype)

    true_pos_predict = math_ops.multiply(y_true, y_pred)  # %6 pos
    # 94% neg...
    true_neg_predict = math_ops.multiply(1.0 - y_true, 1 - y_pred)

    # recision = math_ops.div(correct_pos, predict_pos)
    # recall = math_ops.div(correct_pos, ground_pos)

    # if N_MORE:
    #    m = (2*recall*precision) / (precision+recall)
    # else:
    #    #m = (sensitivity + specificity)/2 # balanced accuracy
    #    raise NotImplementedError("Balanced accuracy metric is not implemented")

    # return m


# 0.5 means no rebalance
def binary_crossentropy_with_focal_seasoned(
    y_true,
    logit_pred,
    beta=FOCAL_LOSS_BETA_NEG_POS,
    gamma=FOCAL_LOSS_GAMMA_NEG_POS,
    alpha=ALPHA,
    custom_weights_in_y_true=True,
):
    """
    :param alpha:weight for positive classes **loss**. default to 1- true positive cnts / all cnts, alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practiceαmay be set by inverse class freqency or hyperparameter.
    :param custom_weights_in_y_true:
    :return:
    """
    balanced = gamma * logit_pred + beta
    y_pred = math_ops.sigmoid(balanced)
    # only use gamma in this layer, easier to split out factor
    return binary_crossentropy_with_focal(
        y_true,
        y_pred,
        gamma=0,
        alpha=alpha,
        custom_weights_in_y_true=custom_weights_in_y_true,
    )


# 0.5 means no rebalance
def binary_crossentropy_with_focal(
    y_true, y_pred, gamma=FOCAL_LOSS_GAMMA, alpha=ALPHA, custom_weights_in_y_true=True
):
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

    if 1.0 - eps <= gamma <= 1.0 + eps:
        bce = alpha * math_ops.multiply(
            1.0 - y_pred, math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        )
        bce += (1 - alpha) * math_ops.multiply(
            y_pred, math_ops.multiply((1.0 - y_true), math_ops.log(1.0 - y_pred + eps))
        )
    elif 0.0 - eps <= gamma <= 0.0 + eps:
        bce = alpha * math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        bce += (1 - alpha) * math_ops.multiply(
            (1.0 - y_true), math_ops.log(1.0 - y_pred + eps)
        )
    else:
        gamma_tensor = tf.broadcast_to(tf.constant(gamma), tf.shape(input=y_pred))
        bce = alpha * math_ops.multiply(
            math_ops.pow(1.0 - y_pred, gamma_tensor),
            math_ops.multiply(y_true, math_ops.log(y_pred + eps)),
        )
        bce += (1 - alpha) * math_ops.multiply(
            math_ops.pow(y_pred, gamma_tensor),
            math_ops.multiply((1.0 - y_true), math_ops.log(1.0 - y_pred + eps)),
        )

    if custom_weights_in_y_true:
        return math_ops.multiply(-bce, custom_weights)
    else:
        return -bce


def main(argv):
    args = parser.parse_args(argv[1:])

    global kernel
    logger.info("Will start load data")
    # kernel = KaggleKernel(action="convert_train_data")
    action = None
    if CONVERT_DATA:
        # action = TRAIN_DATA_EXCLUDE_IDENDITY_ONES
        # action = CONVERT_DATA_Y_NOT_BINARY
        # action = CONVERT_TRAIN_DATA  # given the pickle of numpy train data
        # given the pickle of numpy train data
        action = CONVERT_ADDITIONAL_NONTOXIC_DATA
    else:
        if (
            EXCLUDE_IDENTITY_IN_TRAIN and not IDENTITY_RUN
        ):  # for identity, need to predict for all train data
            action = TRAIN_DATA_EXCLUDE_IDENDITY_ONES
    # if not (RESTART_TRAIN or RESTART_TRAIN_ID or RESTART_TRAIN_RES):
    #    action = DATA_ACTION_NO_NEED_LOAD_EMB_M  # loading model from h5 file, no need load emb matrix (save memory)
    if FINAL_SUBMIT:
        kernel = KaggleKernel(action=None)
    else:
        kernel = KaggleKernel(action=action)
        logger.debug(action)
    kernel.load_identity_data_idx()  # so only predict part of the data

    logger.info("load data done")

    # pred = pickle.load(open('predicts', 'rb'))
    prefix = kernel.emb.BIN_FOLDER

    if IDENTITY_RUN:
        # preds = np.where(preds >= 0.5, True, False) no need recording to API description, but why auc value can change?
        for idtt in d.IDENTITY_COLUMNS:
            # for idtt in ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish','muslim', 'black', 'white', 'psychiatric_or_mental_illness'][6:]:
            if PRD_ONLY:
                # to_predict = kernel.to_predict_X_all
                to_predict_all = utils.get_obj_and_save("additional_data.bin")
                len_all = len(to_predict_all)
                one_len = int(len_all // 10)
                id_additional_pred = []
                for i in range(10):
                    to_predict = kernel.emb._prepare_additional_data_pad(
                        to_predict_all[one_len * i : one_len * (i + 1)]
                    )

                    set_trace()
                    id_pred = kernel.run_identity_model(
                        idtt,
                        to_predict,
                        None,
                        params={
                            "prefix": prefix,
                            "starter_lr": STARTER_LEARNING_RATE,
                            "epochs": 8,
                            "patience": 2,
                            "lr_decay": 0.5,
                            "validation_split": 0.05,
                            "no_check_point": False,  # save check point or not
                            "verbose": 2,
                            "continue_train": False,  # just predict, do not train this time
                            "predict_only": True,
                        },
                    )
                    id_additional_pred = id_additional_pred + id_pred
                    del to_predict
                utils.dump_obj(id_additional_pred, f"{idtt}_additional_pred")
            else:  # train
                kernel.run_identity_model(
                    idtt,
                    kernel.train_X_identity,
                    kernel.train_y_identity,
                    params={
                        "prefix": prefix,
                        "starter_lr": STARTER_LEARNING_RATE,
                        "epochs": 8,
                        "patience": 2,
                        "lr_decay": 0.5,
                        "validation_split": 0.05,
                        "no_check_point": False,  # save check point or not
                        "verbose": 2,
                        "continue_train": False,  # just predict, do not train this time
                    },
                )

        return

    if TARGET_RUN == "res":
        # if not os.path.isfile('predicts'):
        preds = pickle.load(open("predicts", "rb"))
        kernel.calculate_metrics_and_print(preds)
        # improve the model with another data input,
        kernel.res_subgroup("white", preds)

    elif TARGET_RUN == "lstm":
        predict_only = PRD_ONLY
        if not TARGET_RUN_READ_RESULT:
            if ANA_RESULT:
                # preds = pickle.load(open('predicts', 'rb'))
                # sample_weight = kernel.prepare_weight_for_subgroup_balance()
                pass
            else:
                sample_weights = kernel.prepare_weight_for_subgroup_balance()
                # sample_weights_train = sample_weights[kernel.train_mask]
                # pickle.dump(val_mask, open("val_mask", 'wb'))  # only the ones with identity is predicted
                val_mask = np.invert(kernel.train_mask)
                # val..., no need to weight changing
                sample_weights[val_mask] = 1.0

                train_X = None
                val_X = None
                train_y_aux = None
                train_y = None

                if NO_AUX:
                    train_X, val_X, train_y, val_y = kernel.prepare_train_labels(
                        kernel.train_y_all,
                        kernel.train_mask,
                        custom_weights=True,
                        with_aux=False,
                        train_y_aux=None,
                        sample_weights=sample_weights,
                    )
                else:
                    (
                        train_X,
                        val_X,
                        train_y,
                        train_y_aux,
                        val_y,
                        val_y_aux,
                    ) = kernel.prepare_train_labels(
                        kernel.train_y_all,
                        kernel.train_mask,
                        custom_weights=True,
                        with_aux=True,
                        train_y_aux=kernel.train_y_aux_all,
                        sample_weights=sample_weights,
                        fortify_subgroup=None,
                    )
                # sample_weights_train = None  # no need to use in fit, will cooperate to the custom loss function
                logger.debug(f"train with {len(train_X)} data entries")
                logger.debug(f"prefix: {prefix}")
                preds = kernel.run_lstm_model(
                    predict_ones_with_identity=True,
                    final_train=FINAL_SUBMIT,
                    params={
                        "prefix": prefix,
                        # will retrain every time if True,restore will report sensitivity problem now
                        "re-start-train": RESTART_TRAIN,
                        "predict-only": predict_only,
                        "starter_lr": STARTER_LEARNING_RATE,
                        # 'sample_weights': sample_weights_train,
                        # train data with identities
                        "train_data": (train_X, train_y),
                        "train_y_aux": train_y_aux,
                        "val_y_aux": val_y_aux,
                        # train data with identities
                        "val_data": (val_X, val_y),
                        "patience": 2,
                        "binary": True,
                        # 'fortify_subgroup': 'black',
                    },
                    val_mask=val_mask,
                )  # only the val_mask ones is predicted TODO modify val set, to resemble test set
                # if not FINAL_SUBMIT:  # we still need to check this ... for evalutate our model
        else:
            # ans is in self.train_df.loc[val_mask, 'lstm'] = preds
            preds, val_mask = pickle.load(open("pred_val_mask", "rb"))

        # else:
        #    preds = pickle.load(open('predicts', 'rb'))
        # df[df.white & (df.target_orig<0.5) & (df.lstm > 0.5)][['comment_text','lstm','target_orig']].head()
        # kernel.evaluate_model_and_print(preds, 0.55)

        # todo later we could split train/test, to see overfit thing, preds here are all ones with identity, need to
        #  split out the ones is in the training set

        # check if binarify will make difference -> yes, the result is worse
        # pred_target = np.where(preds[0] >= 0.5, True, False)
        # value, bias_metrics = kernel.evaluate_model(pred_target)
        # logger.info(value)
        # logger.info(f"\n{bias_metrics}")
    return


# or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
BALANCE_SCHEME_SUBGROUPS = "target_bucket_same_for_subgroups"
# or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
BALANCE_SCHEME_ACROSS_SUBGROUPS = "more_for_low_score"
BALANCE_SCHEME_AUC = "more_bp_sn"
# balance_scheme_target_splits = 'target_bucket_same_for_target_splits' # or 1->0.8 slop or 0.8->1, or y=-(x-1/2)^2+1/4; just test
# not work, because manual change will corrupt orignial information?
BALANCE_SCHEME_TARGET_SPLITS = "no_target_bucket_extreme_positive"

WEIGHT_TO_Y = True


ANA_RESULT = False
if os.path.isfile(".ana_result"):
    ANA_RESULT = True
    RESTART_TRAIN = False

    IDENTITY_RUN = False
    TARGET_RUN = "lstm"
    PRD_ONLY = True
    RESTART_TRAIN_RES = False

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logger = utils.logger
    logger.info(
        f"Start run with FL_{FOCAL_LOSS}_{FOCAL_LOSS_GAMMA}_{ALPHA} lr {STARTER_LEARNING_RATE}, decay {LEARNING_RATE_DECAY_PER_EPOCH}, \
BS {BATCH_SIZE}, NO_ID_IN_TRAIN {EXCLUDE_IDENTITY_IN_TRAIN}, \
EPOCHS {EPOCHS}, Y_TRAIN_BIN {Y_TRAIN_BIN}\n{BALANCE_SCHEME_SUBGROUPS} {BALANCE_SCHEME_ACROSS_SUBGROUPS} {BALANCE_SCHEME_TARGET_SPLITS} {BALANCE_SCHEME_AUC}"
    )
    main([1])
    # tf.compat.v1.app.run(main)
    logger.info(
        f"Start run with FL_{FOCAL_LOSS}_{FOCAL_LOSS_GAMMA}_{ALPHA} lr {STARTER_LEARNING_RATE}, decay {LEARNING_RATE_DECAY_PER_EPOCH}, \
BS {BATCH_SIZE}, NO_ID_IN_TRAIN {EXCLUDE_IDENTITY_IN_TRAIN}, \
EPOCHS {EPOCHS}, Y_TRAIN_BIN {Y_TRAIN_BIN}\n{BALANCE_SCHEME_SUBGROUPS} {BALANCE_SCHEME_ACROSS_SUBGROUPS} {BALANCE_SCHEME_TARGET_SPLITS} {BALANCE_SCHEME_AUC}"
    )
