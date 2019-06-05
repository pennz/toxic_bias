import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Model, load_model
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate, BatchNormalization
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, PReLU
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ProgbarLogger
from keras.metrics import mean_absolute_error, binary_crossentropy, mean_squared_error
import keras.backend as K

from tensorflow.python.ops import math_ops
from sklearn.model_selection import KFold

#from gradient_reversal_keras_tf.flipGradientTF import GradientReversal

import data_prepare as d
import os
import pickle
import logging

# NUM_MODELS = 2  # might be helpful but...

# BATCH_SIZE = 2048 * 2  # for cloud server runing
BATCH_SIZE = 1024  # for cloud server runing
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4  # 4 seems good for current setting, more training will help for the final score

kernel = None

from keras import initializers, regularizers, constraints

# Credits for https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class AttentionRaffel(Layer):
    def __init__(self, step_dim=d.MAX_LEN,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
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
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionRaffel,self).__init__(**kwargs)

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
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # Input shape 3D tensor with shape: `(samples, steps, features)`.
        # one step is means one bidirection?
        assert len(input_shape) == 3

        self.W = self.add_weight('{}_W'.format(self.name),
                                 (input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]  # features dimention of input

        if self.bias:
            self.b = self.add_weight('{}_b'.format(self.name),
                                     (input_shape[1],),
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


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()


def identity_model(features, labels, mode, params):
    # words = Input(shape=(None,))
    embedding_matrix = params['embedding_matrix']
    identity_out_num = params['identity_out_num']

    words = features  # input layer, so features need to be tensor!!!

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    #### Output Layer
    result = Dense(identity_out_num, activation='sigmoid')(hidden)

    #### Implement training, evaluation, and prediction
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
    loss = tf.keras.losses.binary_crossentropy(labels, result)  # todo put it together, fine?

    # Compute evaluation metrics.
    # m = tf.keras.metrics.SparseCategoricalAccuracy()
    # m.update_state(labels, logits)
    # accuracy = m.result()
    # metrics = {'accuracy': accuracy}
    # tf.compat.v1.summary.scalar('accuracy', accuracy)
    binary_accuracy = tf.keras.metrics.binary_accuracy(labels, result)

    metrics = {'accuracy': binary_accuracy}
    tf.compat.v1.summary.scalar('accuracy', binary_accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op. # will be called by the estimator
    assert mode == tf.estimator.ModeKeys.TRAIN

    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate'])
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.compat.v1.train.get_global_step(),
            decay_steps=params['decay_steps'],
            staircase=True,
            decay_rate=0.5))
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class NBatchProgBarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, count_mode='samples', stateful_metrics=None, display_per_batches=1000, verbose=1,
                 early_stop=False, patience_displays=0, epsilon=1e-7):
        super(NBatchProgBarLogger, self).__init__(count_mode, stateful_metrics)
        self.display_per_batches = display_per_batches
        self.step_idx = 0  # across epochs
        self.display_idx = 0  # across epochs
        self.verbose = verbose

        self.early_stop = early_stop  # better way is subclass EearlyStopping callback.
        self.patience_displays = patience_displays
        self.losses = np.empty(patience_displays, dtype=np.float32)
        self.losses_sum_display = 0
        self.epsilon = epsilon
        self.stopped_step = 0

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
                    f'\nmean: {avg_loss_per_display}, Step {display_info_start_step }({display_info_start_step*BATCH_SIZE}) to {self.step_idx}({self.step_idx*BATCH_SIZE}) for {self.display_idx}th display step')

                self.display_idx += 1  # used in index, so +1 later
                if self.display_idx >= self.patience_displays:
                    std = np.std(
                        self.losses)  # as SGD, always variance, so not a good way, need to learn from early stopping
                    std_start_step = self.step_idx - self.display_per_batches * self.patience_displays + 1
                    print(f'mean: {np.mean(self.losses)}, std:{std} for Step {std_start_step}({std_start_step*BATCH_SIZE}) to {self.step_idx}({self.step_idx*BATCH_SIZE}) for {self.display_idx}th display steps')
                    if std < self.epsilon:
                        self.stopped_step = self.step_idx
                        self.model.stop_training = True
                        print(
                            f'Early Stop criterion met: std is {std} at Step {self.step_idx} for {self.display_idx}th display steps')

    def on_train_end(self, logs=None):
        if self.stopped_step > 0 and self.verbose > 0:
            print('Step %05d: early stopping' % (self.stopped_step + 1))



class KaggleKernel:
    def __init__(self, action=None):
        self.model = None
        self._val_index = None  # for debug
        self.emb = None
        self.train_X = None
        self.train_y = None
        self.train_y_aux = None
        self.train_y_identity = None
        self.train_X_identity = None
        self.to_predict_X = None
        self.embedding_matrix = None
        self.identity_idx = None

        self.oof_preds = None
        self.load_data(action)

    def load_data(self, action):
        if self.emb is None:
            self.emb = d.EmbeddingHandler()
        self.train_X, self.train_y, self.train_y_aux, self.to_predict_X, self.embedding_matrix = self.emb.data_prepare(
            action)
        if Y_TRAIN_BIN:
            self.train_y_float_backup = self.train_y
            self.train_y = np.where(self.train_y > 0.5, True, False)
        if action is not None:
            if action == TRAIN_DATA_EXCLUDE_IDENDITY_ONES:
                self.load_identity_data_idx()
                mask = np.ones(len(self.train_X), np.bool)
                mask[self.identity_idx] = 0  # reverse indexing

                self.train_X = self.train_X[mask]
                self.train_y = self.train_y[mask]
                self.train_y_aux = self.train_y_aux[mask]
                logger.debug("Train data no identity ones now")

        try:
            if self.emb.do_emb_matrix_preparation:
                exit(0)  # saved and can exit
        except:
            logger.warning(
                "Prepare emb for embedding error, so we might already have process file and load data, and we continue")

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
        #if FOCAL_LOSS_GAMMA == 2.0:
        #    threshold = 0.57
        #elif FOCAL_LOSS_GAMMA == 1.0:
        #    threshold = (0.53 + (
        #                0.722 - 0.097)) / 2  # (by...reading the test result..., found it changes every training... so useless)
        threshold = math_ops.cast(threshold, y_pred.dtype)
        y_pred_b = math_ops.cast(y_pred > threshold, y_pred.dtype)
        y_true_b = math_ops.cast(y_true > threshold, y_pred.dtype)

        #ground_pos = math_ops.reduce_sum(y_true) + epsilon
        #correct_pos = math_ops.reduce_sum(math_ops.multiply(y_true, y_pred)) + epsilon
        #predict_pos = math_ops.reduce_sum(y_pred) + epsilon
        true_cnt = math_ops.reduce_sum(y_true_b) + epsilon
        false_cnt = math_ops.reduce_sum(1-y_true_b) + epsilon
        pred_true_cnt = math_ops.reduce_sum(y_pred_b) + epsilon
        pred_false_cnt = math_ops.reduce_sum(1-y_pred_b) + epsilon
        #true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
        #false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

        #true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
        #false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

        #tp_mean_scaled = math_ops.cast(true_predict_mean*100, tf.int8)
        #tp_mean_scaled = math_ops.cast(tp_mean_scaled, tf.float32)
        #precision = math_ops.div(correct_pos, predict_pos)
        #recall = math_ops.div(correct_pos, ground_pos)

        #if N_MORE:
        #    m = (2 * recall * precision) / (precision + recall)
        #else:
        #    # m = (sensitivity + specificity)/2 # balanced accuracy
        #    raise NotImplementedError("Balanced accuracy metric is not implemented")

        return (pred_false_cnt-false_cnt)/false_cnt #(batchsize 1024)

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
        #if FOCAL_LOSS_GAMMA == 2.0:
        #    threshold = 0.57
        #elif FOCAL_LOSS_GAMMA == 1.0:
        #    threshold = (0.53 + (
        #                0.722 - 0.097)) / 2  # (by...reading the test result..., found it changes every training... so useless)
        threshold = math_ops.cast(threshold, y_pred.dtype)
        y_pred_b = math_ops.cast(y_pred > threshold, y_pred.dtype)
        y_true_b = math_ops.cast(y_true > threshold, y_pred.dtype)

        #ground_pos = math_ops.reduce_sum(y_true) + epsilon
        #correct_pos = math_ops.reduce_sum(math_ops.multiply(y_true, y_pred)) + epsilon
        #predict_pos = math_ops.reduce_sum(y_pred) + epsilon
        true_cnt = math_ops.reduce_sum(y_true_b) + epsilon
        false_cnt = math_ops.reduce_sum(1-y_true_b) + epsilon
        pred_true_cnt = math_ops.reduce_sum(y_pred_b) + epsilon
        pred_false_cnt = math_ops.reduce_sum(1-y_pred_b) + epsilon
        #true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
        #false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

        #true_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, y_true_b))/true_cnt
        #false_predict_mean = math_ops.reduce_sum(math_ops.multiply(y_pred, 1-y_true_b))/false_cnt

        #tp_mean_scaled = math_ops.cast(true_predict_mean*100, tf.int8)
        #tp_mean_scaled = math_ops.cast(tp_mean_scaled, tf.float32)
        #precision = math_ops.div(correct_pos, predict_pos)
        #recall = math_ops.div(correct_pos, ground_pos)

        #if N_MORE:
        #    m = (2 * recall * precision) / (precision + recall)
        #else:
        #    # m = (sensitivity + specificity)/2 # balanced accuracy
        #    raise NotImplementedError("Balanced accuracy metric is not implemented")

        return (pred_true_cnt-true_cnt)/true_cnt #(batchsize 1024)

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

    def build_identity_model(self, identity_out_num):  # so nine model for them...
        # d.IDENTITY_COLUMNS  # with more than 500
        """build lstm model

        :param embedding_matrix:
        :param num_aux_targets:
        :return: the built model lstm
        """
        words = Input(shape=(None,))
        x = Embedding(*self.embedding_matrix.shape, weights=[self.embedding_matrix], trainable=False)(words)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(CuDNNLSTM(int(LSTM_UNITS // 2), return_sequences=True))(x)

        hidden = concatenate([
            GlobalMaxPooling1D()(x),
            GlobalAveragePooling1D()(x),
        ])
        hidden = add([hidden, Dense(int(DENSE_HIDDEN_UNITS // 2), activation='relu')(hidden)])
        result = Dense(identity_out_num, activation='sigmoid')(hidden)
        # result_with_aux = Dense(num_aux_targets+1, activation='sigmoid')(hidden)

        model = Model(inputs=words, outputs=result)
        # model = Model(inputs=words, outputs=result_with_aux)
        # model.compile(loss='binary_crossentropy', optimizer='adam')

        model.compile(loss='binary_crossentropy', optimizer=Adam(0.005))  # need to improve#for target it should be fine, they are positive related
        return model

    def run_identity_model(self, n_splits, features, labels, params):

        splits = list(KFold(n_splits=n_splits, random_state=2019, shuffle=True).split(
            features))  # just use sklearn split to get id and it is fine. For text thing,
        # memory is enough, fit in, so tfrecord not needed, just pickle and load it all to memory
        # TODO condiser use train test split as, so no need to run 5 times...
        '''
        train_df, validate_df = model_selection.train_test_split(train, test_size=0.2)
        logger.info('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))'''
        prefix = params['prefix']

        # self.oof_preds = np.zeros(labels.shape)
        # test_preds = np.zeros((self.to_predict_X.shape[0]))
        for i, identity in enumerate(d.IDENTITY_COLUMNS):  # not sure the index is right or not, from pandas to numpy
            for fold in range(n_splits):
                K.clear_session()  # so it will start over, across different fold, but we will load model form checkpoint
                tr_ind, val_ind = splits[fold]
                file_name = prefix + '_' + identity + f'_{fold}.hdf5'

                ckpt = ModelCheckpoint(file_name, save_best_only=True, verbose=1)
                early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

                if not os.path.isfile(file_name):
                    model = self.build_identity_model(1)  # one model per identity...
                    # model = self.build_identity_model(len(d.IDENTITY_COLUMNS))
                else:
                    logger.info("restore from the model file")
                    model = load_model(file_name,
                                       custom_objects={'AttentionRaffel':AttentionRaffel})
                    logger.info("restore from the model file -> done\n\n\n\n\n")

                self.model = model  # for debug
                model.fit(features[tr_ind],
                          # self.train_y[tr_ind]>0.5,
                          labels[tr_ind][:, i],
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          verbose=0,
                          # validation_data=(self.train_X[val_ind], self.train_y[val_ind]>0.5),
                          validation_data=(features[val_ind], labels[val_ind][:, i]),
                          callbacks=[
                              LearningRateScheduler(
                                  lambda e: STARTER_LEARNING_RATE * (LEARNING_RATE_DECAY_PER_EPOCH ** e),
                                  verbose=1
                              ),
                              early_stop,
                              ckpt
                          ])
                break  # only run one fold...

            # schedule: a function that takes an epoch index as input(integer, indexed from 0) and current learning rate and returns a new learning rate as output(float).
            # verbose: int. 0: quiet, 1: update messages.

            # self._val_index = val_ind  # for debug
            # pred = model.predict(self.train_X[val_ind], verbose=2)
            # self.oof_preds[val_ind] += pred

        # logger.info('Overall predict, binary accuracy')
        # logger.info(K.get_session().run(KaggleKernel.bin_acc_2(labels, self.oof_preds)).mean())
        # logger.info('Overall predict, binary accuracy, compare with all zero')
        # logger.info(K.get_session().run(KaggleKernel.bin_acc_2(labels, np.zeros(labels.shape))).mean())
        """:(
Overall predict, binary accuracy
0.930945
Overall predict, binary accuracy, compare with all zero
0.9437693"""

        pickle.dump(self.oof_preds, open(prefix + "predicts", 'wb'))

    def build_lstm_model_customed(self, num_aux_targets, with_aux=False, loss='binary_crossentropy', metrics=None,
                                  hidden_act='relu', with_BN=False):
        """build lstm model, non-binarized

        cls_reg: 0 for classification, 1 for regression(linear)
        metrics: should be a list
        with_aux: default False, don't know how to interpret...
        with_BN: ... BatchNormalization not work, because different subgroup different info, batch normalize will make it worse?
        """
        logger.debug(f'model detail: loss {loss}, hidden_act {hidden_act}, with_BN {with_BN}')
        words = Input(shape=(d.MAX_LEN,))  # (None, 180)
        x = Embedding(*self.embedding_matrix.shape, weights=[self.embedding_matrix], trainable=False)(words)

        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

        hidden = concatenate([
            AttentionRaffel(d.MAX_LEN, name="attention_after_lstm")(x),
            GlobalMaxPooling1D()(x),
            #GlobalAveragePooling1D()(x),
        ])

        activate_type = hidden_act
        if activate_type == 'prelu':  # found it not working
            hidden = add([hidden, PReLU()(Dense(DENSE_HIDDEN_UNITS, activation=None)(hidden))])
            if with_BN: hidden = BatchNormalization()(hidden)
            hidden = add([hidden, PReLU()(Dense(DENSE_HIDDEN_UNITS, activation=None)(hidden))])
        else:
            hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation=activate_type)(hidden)])
            if with_BN: hidden = BatchNormalization()(hidden)
            hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation=activate_type)(hidden)])

        result = Dense(1, activation='sigmoid')(hidden)

        if with_aux:
            aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
            model = Model(inputs=words, outputs=[result, aux_result])
        else:
            model = Model(inputs=words, outputs=result)

        model.compile(loss=loss, optimizer=Adam(0.005, amsgrad=True), metrics=metrics)

        return model

    def build_lstm_model(self, num_aux_targets):
        words = Input(shape=(None,))
        x = Embedding(*self.embedding_matrix.shape, weights=[self.embedding_matrix], trainable=False)(words)
        logger.info("Embedding fine, here the type of embedding matrix is {}".format(type(self.embedding_matrix)))

        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

        hidden = concatenate([
            GlobalMaxPooling1D()(x),
            GlobalAveragePooling1D()(x),
        ])
        hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
        hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
        result = Dense(1, activation='sigmoid')(hidden)
        aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

        model = Model(inputs=words, outputs=[result, aux_result])
        # model = Model(inputs=words, outputs=result_with_aux)
        # model.compile(loss='binary_crossentropy', optimizer='adam')

        model.compile(loss='binary_crossentropy', optimizer=Adam(0.005), metrics=[KaggleKernel.bin_prd_clsf_info])
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

    def run_lstm_model(self, final_train=False, n_splits=5, predict_ones_with_identity=True, train_test_split=True,
                       params=None):
        # checkpoint_predictions = []
        # weights = []
        splits = list(KFold(n_splits=n_splits, random_state=2019, shuffle=True).split(
            self.train_X))  # just use sklearn split to get id and it is fine. For text thing,
        # memory is enough, fit in, so tfrecord not needed, just pickle and load it all to memory
        # TODO condiser use train test split as
        '''
        train_df, validate_df = model_selection.train_test_split(train, test_size=0.2)
        logger.info('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))'''

        self.oof_preds = np.zeros((self.train_X.shape[0], 1 + self.train_y_aux.shape[1]))
        # test_preds = np.zeros((self.to_predict_X.shape[0]))
        prefix = params['prefix']
        re_train = params['re-start-train']
        predict_only = params['predict-only']

        prefix += 'G{:.1f}'.format(FOCAL_LOSS_GAMMA)

        for fold in range(n_splits):
            K.clear_session()  # so it will start over
            tr_ind, val_ind = splits[fold]

            if NO_AUX:
                h5_file = prefix + '_attention_lstm_NOAUX_' + f'{fold}.hdf5'  # we could load with file name, then remove and save to new one
            else:
                h5_file = prefix + '_attention_lstm_' + f'{fold}.hdf5'  # we could load with file name, then remove and save to new one

            ckpt = ModelCheckpoint(h5_file, save_best_only=True, verbose=1)
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)

            starter_lr = STARTER_LEARNING_RATE
            # model thing
            if re_train or not os.path.isfile(h5_file):
                if NO_AUX:
                    if FOCAL_LOSS:
                        model = self.build_lstm_model_customed(0, with_aux=False,
                                                               loss=binary_crossentropy_with_focal,
                                                               metrics=[#tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.SpecificityAtSensitivity(0.50),
                                                                      binary_crossentropy, #tf.keras.metrics.Mean(),
                                                                        mean_absolute_error,])
                                                                      #tf.keras.metrics.SensitivityAtSpecificity(0.9, name='sn_90'),
                                                                      #tf.keras.metrics.SensitivityAtSpecificity(0.95, name='sn_95'),
                                                                      #tf.keras.metrics.SpecificityAtSensitivity(0.90, name="sp_90"),
                                                                      #tf.keras.metrics.SpecificityAtSensitivity(0.95, name="sp_95"),
#                        KaggleKernel.bin_prd_clsf_info_neg,
#                        KaggleKernel.bin_prd_clsf_info_pos])
                                                                           #tf.keras.metrics.SpecificityAtSensitivity(0.98), tf.keras.metrics.SpecificityAtSensitivity(0.99), tf.keras.metrics.SpecificityAtSensitivity(1.00)
                        # Sensitivity measures the proportion of actual positives that are correctly identified as such (tp / (tp + fn)).
                    else:
                        model = self.build_lstm_model_customed(0, with_aux=False)
                else:
                    model = self.build_lstm_model(len(self.train_y_aux[0]))
                self.model = model
                logger.info('build model -> done')

            else:
                model = load_model(h5_file, custom_objects={'binary_crossentropy_with_focal': binary_crossentropy_with_focal, 'AttentionRaffel':AttentionRaffel})
                starter_lr = starter_lr * LEARNING_RATE_DECAY_PER_EPOCH ** EPOCHS
                self.model = model
                logger.debug('restore from the model file {} -> done'.format(h5_file))

            # data thing
            if NO_AUX:
                y_train = self.train_y[tr_ind]
                y_val = self.train_y[val_ind]
            else:
                y_train = [self.train_y[tr_ind], self.train_y_aux[tr_ind]]
                y_val = [self.train_y[val_ind], self.train_y_aux[val_ind]]

            # model.fit(self.train_X[tr_ind],
            #          #self.train_y[tr_ind]>0.5,
            #          y_train,
            #          batch_size=BATCH_SIZE,
            #          epochs=EPOCHS,
            #          #steps_per_epoch=int(len(tr_ind)/BATCH_SIZE),
            #          verbose=0,
            #          #validation_data=(self.train_X[val_ind], self.train_y[val_ind]>0.5),
            #          validation_data=(self.train_X[val_ind], y_val),
            #          callbacks=[
            #              LearningRateScheduler(
            #                  #STARTER_LEARNING_RATE = 1e-2
            #                  #LEARNING_RATE_DECAY_PER_EPOCH = 0.7
            #                  lambda e: starter_lr * (LEARNING_RATE_DECAY_PER_EPOCH ** e),
            #                  verbose=1
            #              ),
            #              early_stop,
            #              ckpt,
            #              tf_fit_batch_logger])
            if predict_only:
                break  # do not fit
            model.fit(self.train_X,
                      # self.train_y[tr_ind]>0.5,
                      self.train_y,
                      validation_split=0.05,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      # steps_per_epoch=int(len(self.train_X)*0.95/BATCH_SIZE),
                      verbose=0,
                      # validation_data=(self.train_X[val_ind], self.train_y[val_ind]>0.5),
                      # validation_data=(self.train_X[val_ind], y_val),
                      callbacks=[
                          LearningRateScheduler(
                              # STARTER_LEARNING_RATE = 1e-2
                              # LEARNING_RATE_DECAY_PER_EPOCH = 0.7
                              lambda e: starter_lr * (LEARNING_RATE_DECAY_PER_EPOCH ** e),
                              verbose=1
                          ),
                          early_stop,
                          ckpt,
                          # tf_fit_batch_logger,
                          prog_bar_logger
                      ])


            if train_test_split:
                break

            # schedule: a function that takes an epoch index as input(integer, indexed from 0) and current learning rate and returns a new learning rate as output(float).
            # verbose: int. 0: quiet, 1: update messages.

            # self._val_index = val_ind  # for debug
            pred = model.predict(self.train_X[val_ind], verbose=1, batch_size=BATCH_SIZE)
            self.oof_preds[val_ind] += np.concatenate((pred[0], pred[1]), axis=1)  # target and aux

        if predict_ones_with_identity:
            return model.predict(self.train_X_identity, verbose=2, batch_size=BATCH_SIZE)

            # test_preds += (np.array(model.predict(self.to_predict_X, verbose=1)[0])).ravel()
        # test_preds /= 5
        # self.preds = test_preds

        # self.save_result(test_preds)

        from sklearn.metrics import roc_auc_score
        # it can Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # roc_auc_score(self.train_y>0.5, oof_preds)
        # logger.info("Will start tf data handling")
        # train_y_aux = np.concatenate((np.expand_dims(train_y, axis=1), train_y_aux), axis=1)
        # global _d
        # global _dd # for debug

        # _d = train_y_aux
        # ds = tf.data.Dataset.from_tensor_slices(train_y_aux)
        # _dd = ds
        # filename = 'train_X_data.tfrecord'
        # writer = tf.data.experimental.TFRecordWriter(filename)
        # writer.write(ds)
        # logger.info("done save tfrecord")

        # train_data = train_input_fn_general(train_X, train_y_aux, args.batch_size, True, split_id=fold, n_splits=5, handle_large=True, shuffle_size=1000, repeat=False)
        # val_data = eval_input_fn_general(train_X, train_y_aux, args.batch_size, True, split_id=fold, n_splits=5, handle_large=True) # for eval, no need to shuffle
        # logger.info("after tf data handling")

    def save_result(self, predictions):
        submission = pd.DataFrame.from_dict({
            'id': self.emb.test_df_id,
            # 'id': test_df.id,
            'prediction': predictions
        })
        submission.to_csv('submission.csv', index=False)

    def filter_out_second_stage_data(self, y_pred, subgroup='white'):
        id_df = self.emb.get_identity_df()
        id_df[subgroup].index  # index for this subgroup
        self.train_y

    def prepare_identity_data(self):
        if self.train_y_identity is None:
            self.train_X_identity, self.train_y_identity = self.emb.get_identity_train_data()
        return self.train_X_identity, self.train_y_identity
        # features = self.train_X
        # labels = self.train_y_identity
        # features_placeholder = tf.placeholder(features.dtype, features.shape)
        # labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        # ds = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        # iterator = ds.make_initializable_iterator()
        # next_element = iterator.get_next()

        # sess = K.get_session()
        # itr = sess.run(iterator.initializer, feed_dict={
        #    features_placeholder: features,
        #    labels_placeholder: labels})
        ##i = 0
        ##while i < 10:
        ##    try:
        ##        el = sess.run(next_element)
        ##        logger.info("should be fine to enumerate")
        ##        logger.info(el)
        ##        i += 1
        ##    except tf.errors.OutOfRangeError:
        ##        break
        # return itr

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
            self.train_X_identity, self.train_y_identity, self.identity_idx = self.emb.get_identity_train_data_df_idx()  # to train the identity
        # return self.train_X_identity, self.train_y_identity

    def evaluate_model(self, preds, threshold=0.5):

        self.emb.read_csv(train_only=True)

        self.load_identity_data_idx()
        self.judge = d.BiasBenchmark(kernel.emb.train_df.iloc[self.identity_idx],threshold=threshold)  # the idx happen to be the iloc value
        # and this judge, read data from pandas df, won't be affected by our operation to train_X. which is good
        #value, bias_metrics, dist, overall_dist = kernel.judge.calculate_benchmark(preds)  # preds should be the same size
        return kernel.judge.calculate_benchmark(preds)  # preds should be the same size

    def evaluate_model_and_print(self, preds, threshold=0.5):
        value, bias_metrics, subgroup_distribution, overall_distribution = self.evaluate_model(preds, threshold)

        bias_metrics_df = bias_metrics.set_index('subgroup')

        pickle.dump(bias_metrics_df, open("bias_metrics", 'wb'))  # only the ones with identity is predicted
        pickle.dump(subgroup_distribution, open("subgroup_dist", 'wb'))  # only the ones with identity is predicted

        # bias_metrics_df = pickle.load(open("bias_metrics", 'rb'))  # only the ones with identity is predicted
        # subgroup_distribution = pickle.load(open("subgroup_dist", 'rb'))  # only the ones with identity is predicted

        logger.info(f'final metric: {value} for threshold {threshold} applied to \'{d.TARGET_COLUMN}\' column')
        logger.info("\n{}".format(bias_metrics[['subgroup', 'subgroup_auc', 'bnsp_auc', 'bpsn_auc']]))
        # logger.info(subgroup_distribution)
        print("### subgroup auc")
        for d0 in subgroup_distribution:
            g = d0['subgroup']
            m = 'subgroup_auc'
            s = 'subgroup_size'
            auc = "{0:.4} {1}".format(bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s])
            print("{0:5.5} ".format(g) + auc + '\t' + str(d0[m][2]) + '\t' + str(d0[m][3]))

        print("### bpsn auc")
        for d0 in subgroup_distribution:
            g = d0['subgroup']
            m = 'bpsn_auc'
            s = 'subgroup_size'
            auc = "{0:.4} {1}".format(bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s])
            print("{0:5.5} ".format(g) + auc + '\t' + str(d0[m][2]) + '\t' + str(d0[m][3]))

        print("### bnsp auc")
        for d0 in subgroup_distribution:
            g = d0['subgroup']
            m = 'bnsp_auc'
            s = 'subgroup_size'
            auc = "{0:.4} {1}".format(bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s])
            print("{0:5.5} ".format(g) + auc + '\t' + str(d0[m][2]) + '\t' + str(d0[m][3]))

        print("### counts")
        # length thing
        for d0 in subgroup_distribution:
            g = d0['subgroup']
            m = 'subgroup_auc'
            s = 'subgroup_size'
            auc = "{0:.4} {1}".format(bias_metrics_df.loc[g][m], bias_metrics_df.loc[g][s])
            print("{0:5.5} ".format(g) + auc + '\t' + str(d0[m][0]) + '\t' + str(d0[m][1]))

        print("### overall")
        g = 'overall'
        m = d.OVERALL_AUC
        s = 'subgroup_size'
        auc = "{0:.4} {1}".format(overall_distribution[m], overall_distribution[s])
        dist = overall_distribution['distribution']
        print(f'{g:5.5} {auc}\tneg_tgt_pred_dis:{dist[2]}\tpos_tgt_pred_dis:{dist[3]}\noverall_pos_neg_cnt:\t{dist[0]}')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_steps', default=2, type=int,
                    help='number of training steps')
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='learing rate')

# lambda e: 5e-3 * (0.5 ** e),
STARTER_LEARNING_RATE = 5e-3 # as the BCE we adopted...
LEARNING_RATE_DECAY_PER_EPOCH = 0.5

IDENTITY_RUN = False
TARGET_RUN = True
PRD_ONLY = True
RESTART_TRAIN = False

NO_AUX = True
Y_TRAIN_BIN = False  # with True, slightly worse
#tf.keras.metrics.SpecificityAtSensitivity(0.50), TRAIN_BIN need to be as we use this metrics, or we can customize
FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.
ALPHA = 0.666
#GAMMA works better 2. with BS 1024
#GAMMA works better 1.5 with BS 512

# _debug_train_data = None

CONVERT_DATA = False
CONVERT_DATA_Y_NOT_BINARY = 'convert_data_y_not_binary'
CONVERT_TRAIN_DATA = 'convert_train_data'  # given the pickle of numpy train data

EXCLUDE_IDENTITY_IN_TRAIN = True
TRAIN_DATA_EXCLUDE_IDENDITY_ONES = 'TRAIN_DATA_EXCLUDE_IDENDITY_ONES'

NEG_RATIO = (1 - 0.05897253769515213)


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
    ture_neg_predict = math_ops.multiply(1. - y_true, y_pred)  # 94% neg...

    # recision = math_ops.div(correct_pos, predict_pos)
    # recall = math_ops.div(correct_pos, ground_pos)

    # if N_MORE:
    #    m = (2*recall*precision) / (precision+recall)
    # else:
    #    #m = (sensitivity + specificity)/2 # balanced accuracy
    #    raise NotImplementedError("Balanced accuracy metric is not implemented")

    # return m


def binary_crossentropy_with_focal(y_true, y_pred, gamma=FOCAL_LOSS_GAMMA, alpha=ALPHA):  # 0.5 means no rebalance
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

    if 1. - eps <= gamma <= 1. + eps:
        bce = alpha * math_ops.multiply(1. - y_pred, math_ops.multiply(y_true, math_ops.log(y_pred + eps)))
        bce += (1 - alpha) * math_ops.multiply(y_pred,
                                               math_ops.multiply((1. - y_true), math_ops.log(1. - y_pred + eps)))
    else:
        gamma_tensor = tf.broadcast_to(tf.constant(gamma), tf.shape(input=y_pred))
        bce = alpha * math_ops.multiply(math_ops.pow(1. - y_pred, gamma_tensor),
                                        math_ops.multiply(y_true, math_ops.log(y_pred + eps)))
        bce += (1 - alpha) * math_ops.multiply(math_ops.pow(y_pred, gamma_tensor),
                                               math_ops.multiply((1. - y_true), math_ops.log(1. - y_pred + eps)))

    return -bce


def main(argv):
    args = parser.parse_args(argv[1:])

    global kernel
    logger.info("Will start load data")
    # kernel = KaggleKernel(action="convert_train_data")
    action = None
    if CONVERT_DATA:
        # action = TRAIN_DATA_EXCLUDE_IDENDITY_ONES
        action = CONVERT_DATA_Y_NOT_BINARY
    else:
        if EXCLUDE_IDENTITY_IN_TRAIN:
            action = TRAIN_DATA_EXCLUDE_IDENDITY_ONES
    kernel = KaggleKernel(action=action)
    logger.debug(action)
    kernel.load_identity_data_idx()  # so only predict part of the data

    logger.info("load data done")

    # pred = pickle.load(open('predicts', 'rb'))
    # preds = pred[identity_idx, 0]

    if IDENTITY_RUN:
        # preds = np.where(preds >= 0.5, True, False) no need recording to API description, but why auc value can change?
        kernel.run_identity_model(5, kernel.train_X_identity, kernel.train_y_identity, params={
            'prefix': "/proc/driver/nvidia/"
        })

    if TARGET_RUN:
        # if not os.path.isfile('predicts'):
        predict_only = PRD_ONLY
        preds = kernel.run_lstm_model(predict_ones_with_identity=True, params={
            'prefix': "/proc/driver/nvidia/",
            're-start-train': RESTART_TRAIN, # will retrain every time if True,restore will report sensitivity problem now
            'predict-only': predict_only
        })
        kernel.model  # improve the model with another data input,

        if NO_AUX:
            pickle.dump(preds, open("predicts", 'wb'))  # only the ones with identity is predicted
        else:
            pickle.dump(preds[0], open("predicts", 'wb'))  # only the ones with identity is predicted
            preds = preds[0]
        # else:
        #    preds = pickle.load(open('predicts', 'rb'))
        kernel.evaluate_model_and_print(preds, 0.5)
        #kernel.evaluate_model_and_print(preds, 0.55)

        # todo later we could split train/test, to see overfit thing, preds here are all ones with identity, need to
        #  split out the ones is in the training set

        # check if binarify will make difference -> yes, the result is worse
        # pred_target = np.where(preds[0] >= 0.5, True, False)
        # value, bias_metrics = kernel.evaluate_model(pred_target)
        # logger.info(value)
        # logger.info(f"\n{bias_metrics}")

    # kernel.run_lstm_model()

    # kernel.run_identity_model(5, train_id_X, train_id_y, params={
    #    'prefix': "/proc/driver/nvidia/identity"
    # })
    return


prog_bar_logger = NBatchProgBarLogger(display_per_batches=int(1000000 / 30 / BATCH_SIZE), early_stop=True,
                                      patience_displays=3)

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logger.info(
        f'Start run with lr {STARTER_LEARNING_RATE}, decay {LEARNING_RATE_DECAY_PER_EPOCH}, gamma {FOCAL_LOSS_GAMMA}, \
BS {BATCH_SIZE}, NO_ID_IN_TRAIN {EXCLUDE_IDENTITY_IN_TRAIN}, \
EPOCHS {EPOCHS}, Y_TRAIN_BIN {Y_TRAIN_BIN} ALPHA{ALPHA}')
    main([1])
    # tf.compat.v1.app.run(main)
    logger.info(
        f'Start run with lr {STARTER_LEARNING_RATE}, decay {LEARNING_RATE_DECAY_PER_EPOCH}, gamma {FOCAL_LOSS_GAMMA}, \
BS {BATCH_SIZE}, NO_ID_IN_TRAIN {EXCLUDE_IDENTITY_IN_TRAIN}, \
EPOCHS {EPOCHS}, Y_TRAIN_BIN {Y_TRAIN_BIN} ALPHA{ALPHA}')
