import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from tensorflow.keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.ops import math_ops
from sklearn.model_selection import KFold

import data_prepare as d
import os

#NUM_MODELS = 2  # might be helpful but...

BATCH_SIZE = 2048 * 2  # for cloud server runing
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4

kernel = None

def identity_model(features, labels, mode, params):
    #words = Input(shape=(None,))
    embedding_matrix = params['embedding_matrix']
    identity_out_num = params['identity_out_num']

    words = features  #input layer, so features need to be tensor!!!

    print(type(words))
    print(type(words))
    print(type(words))

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
    #predicted_classes = tf.argmax(input=logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        #predictions = {
        #    'class_ids': predicted_classes[:, tf.newaxis],
        #    'probabilities': tf.nn.softmax(logits),
        #    'logits': logits,
        #}
        return tf.estimator.EstimatorSpec(mode, predictions=result.out)

    # Compute loss.
    loss = tf.keras.losses.binary_crossentropy(labels, result) # todo put it together, fine?

    # Compute evaluation metrics.
    #m = tf.keras.metrics.SparseCategoricalAccuracy()
    #m.update_state(labels, logits)
    #accuracy = m.result()
    #metrics = {'accuracy': accuracy}
    #tf.compat.v1.summary.scalar('accuracy', accuracy)
    binary_accuracy = tf.keras.metrics.binary_accuracy(labels, result)

    metrics = {'accuracy': binary_accuracy}
    tf.compat.v1.summary.scalar('accuracy', binary_accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op. # will be called by the estimator
    assert mode == tf.estimator.ModeKeys.TRAIN

    #optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate'])
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.compat.v1.train.get_global_step(),
            decay_steps=params['decay_steps'],
            staircase=True,
            decay_rate=0.5))
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class KaggleKernel:
    def __init__(self):
        self.model = None
        self._val_index = None  # for debug
        self.emb = None
        self.train_X = None
        self.train_y = None
        self.train_y_aux = None
        self.train_y_identity = None
        self.to_predict_X = None
        self.embedding_matrix = None

        self.oof_preds = None

    def load_data(self):
        if self.emb is None:
            self.emb = d.EmbeddingHandler()
        self.train_X, self.train_y, self.train_y_aux, self.to_predict_X, self.embedding_matrix = self.emb.load_data()
        try:
            if self.emb.prepare_emb:
                exit(0) # saved and can exit
        except:
            print("Prepare emb error, so loading data might be fine, and we continue")

    @staticmethod
    def binary_accuracy_both_thr(y_true, y_pred, threshold=0.5):
        threshold = math_ops.cast(threshold, y_pred.dtype)
        y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
        y_true = math_ops.cast(y_true > threshold, y_pred.dtype)
        return K.mean(math_ops.equal(y_true, y_pred), axis=-1)

    def build_identity_model(self, identity_out_num):
        #d.IDENTITY_COLUMNS  # with more than 500
        """build lstm model

        :param embedding_matrix:
        :param num_aux_targets:
        :return: the built model lstm
        """
        words = Input(shape=(None,))
        x = Embedding(*self.embedding_matrix.shape, weights=[self.embedding_matrix], trainable=False)(words)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

        hidden = concatenate([
            GlobalMaxPooling1D()(x),
            GlobalAveragePooling1D()(x),
        ])
        hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
        result = Dense(identity_out_num, activation='sigmoid')(hidden)
        # result_with_aux = Dense(num_aux_targets+1, activation='sigmoid')(hidden)

        model = Model(inputs=words, outputs=result)
        # model = Model(inputs=words, outputs=result_with_aux)
        #model.compile(loss='binary_crossentropy', optimizer='adam')

        model.compile(loss='binary_crossentropy', optimizer=Adam(0.005), metrics=[KaggleKernel.binary_accuracy_both_thr])
        return model


    def run_identity_model(self, n_splits):

        splits = list(KFold(n_splits=n_splits, random_state=2019, shuffle=True).split(self.train_X)) # just use sklearn split to get id and it is fine. For text thing,
        # memory is enough, fit in, so tfrecord not needed, just pickle and load it all to memory
        # TODO condiser use train test split as
        '''
        train_df, validate_df = model_selection.train_test_split(train, test_size=0.2)
        print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))'''

        self.oof_preds = np.zeros((self.train_X.shape[0], 1+self.train_y_aux.shape[1]))
        test_preds = np.zeros((self.to_predict_X.shape[0]))

        for fold in range(n_splits):
            K.clear_session() # so it will start over?
            tr_ind, val_ind = splits[fold]

            ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only=True, verbose=1)
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

            model = self.build_identity_model(len(d.IDENTITY_COLUMNS))
            self.model = model # for debug
            model.fit(self.train_X[tr_ind],
                      #self.train_y[tr_ind]>0.5,
                      [self.train_y[tr_ind], self.train_y_aux[tr_ind]],
                      batch_size=BATCH_SIZE,
                      epochs=1,
                      verbose=1,
                      #validation_data=(self.train_X[val_ind], self.train_y[val_ind]>0.5),
                      validation_data=(self.train_X[val_ind], [self.train_y[val_ind], self.train_y_aux[val_ind]]),
                      callbacks=[
                          LearningRateScheduler(
                              lambda e: 5e-3 * (0.5 ** e),
                              verbose=2
                          ),
                          early_stop,
                          ckpt])

            # schedule: a function that takes an epoch index as input(integer, indexed from 0) and current learning rate and returns a new learning rate as output(float).
            # verbose: int. 0: quiet, 1: update messages.

            #self._val_index = val_ind  # for debug
            pred = model.predict(self.train_X[val_ind], verbose=1)
            self.oof_preds[val_ind] += np.concatenate((pred[0], pred[1]), axis=1)

        import pickle
        pickle.dump(self.oof_preds, open("predicts", 'wb'))

    def build_model(self, num_aux_targets):
        """build lstm model

        :param embedding_matrix:
        :param num_aux_targets:
        :return: the built model lstm

    __________________________________________________________________________________________________
    Layer (type) Output Shape Param # Connected to
    ==================================================================================================
    input_2 (InputLayer) (None, None) 0
    __________________________________________________________________________________________________
    embedding_1 (Embedding) (None, None, 600) 197034000 input_2[0][0]
    __________________________________________________________________________________________________
    spatial_dropout1d_1 (SpatialDro (None, None, 600) 0 embedding_1[0][0]
    __________________________________________________________________________________________________
    bidirectional_2 (Bidirectional) (None, None, 256) 747520 spatial_dropout1d_1[0][0]
    __________________________________________________________________________________________________
    bidirectional_3 (Bidirectional) (None, None, 256) 395264 bidirectional_2[0][0]
    __________________________________________________________________________________________________
    global_max_pooling1d_1 (GlobalM (None, 256) 0 bidirectional_3[0][0]
    __________________________________________________________________________________________________
    global_average_pooling1d_1 (Glo (None, 256) 0 bidirectional_3[0][0]
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate) (None, 512) 0 global_max_pooling1d_1[0][0] global_average_pooling1d_1[0][0]
    __________________________________________________________________________________________________
    dense_4 (Dense) (None, 512) 262656 concatenate_1[0][0]
    __________________________________________________________________________________________________
    add_2 (Add) (None, 512) 0 concatenate_1[0][0] dense_4[0][0]
    __________________________________________________________________________________________________
    dense_5 (Dense) (None, 512) 262656 add_2[0][0]
    __________________________________________________________________________________________________
    add_3 (Add) (None, 512) 0 add_2[0][0] dense_5[0][0]
    __________________________________________________________________________________________________
    dense_6 (Dense) (None, 1) 513 add_3[0][0]
    __________________________________________________________________________________________________
    dense_7 (Dense) (None, 6) 3078 add_3[0][0]
    ==================================================================================================
    Total params: 198,705,687 Trainable params: 1,671,687 Non-trainable params: 197,034,000
    __________________________________________________________________________________________________
        """
        words = Input(shape=(None,))
        x = Embedding(*self.embedding_matrix.shape, weights=[self.embedding_matrix], trainable=False)(words)
        print("Embedding fine, here")
        print(type(self.embedding_matrix))
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
        #model.compile(loss='binary_crossentropy', optimizer='adam')

        model.compile(loss='binary_crossentropy', optimizer=Adam(0.005), metrics=[KaggleKernel.binary_accuracy_both_thr])
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


    def lstm_model_run(self, final_train=False, n_splits=5):
        #checkpoint_predictions = []
        #weights = []
        splits = list(KFold(n_splits=n_splits, random_state=2019, shuffle=True).split(self.train_X)) # just use sklearn split to get id and it is fine. For text thing,
        # memory is enough, fit in, so tfrecord not needed, just pickle and load it all to memory
        # TODO condiser use train test split as
        '''
        train_df, validate_df = model_selection.train_test_split(train, test_size=0.2)
        print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))'''

        self.oof_preds = np.zeros((self.train_X.shape[0], 1+self.train_y_aux.shape[1]))
        test_preds = np.zeros((self.to_predict_X.shape[0]))

        for fold in range(n_splits):
            K.clear_session() # so it will start over
            tr_ind, val_ind = splits[fold]

            h5_file = f'gru_{fold}.hdf5'  # we could load with file name, then remove and save to new one

            ckpt = ModelCheckpoint(h5_file, save_best_only=True, verbose=1)
            early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
            if not os.path.isfile(f'gru_{fold}.hdf5'):
                model = self.build_model(len(self.train_y_aux[0]))
            else:
                print("restore from the model file")
                model = load_model(h5_file,
                                   custom_objects={'binary_accuracy_both_thr': KaggleKernel.binary_accuracy_both_thr})
                print("restore from the model file -> done")
            self.model = model
            model.fit(self.train_X[tr_ind],
                      #self.train_y[tr_ind]>0.5,
                      [self.train_y[tr_ind], self.train_y_aux[tr_ind]],
                      batch_size=BATCH_SIZE,
                      epochs=1,
                      verbose=1,
                      #validation_data=(self.train_X[val_ind], self.train_y[val_ind]>0.5),
                      validation_data=(self.train_X[val_ind], [self.train_y[val_ind], self.train_y_aux[val_ind]]),
                      callbacks=[
                          LearningRateScheduler(
                              lambda e: 5e-3 * (0.5 ** e),
                              verbose=2
                          ),
                          early_stop,
                          ckpt])

            # schedule: a function that takes an epoch index as input(integer, indexed from 0) and current learning rate and returns a new learning rate as output(float).
            # verbose: int. 0: quiet, 1: update messages.

            #self._val_index = val_ind  # for debug
            pred = model.predict(self.train_X[val_ind], verbose=1)
            self.oof_preds[val_ind] += np.concatenate((pred[0], pred[1]), axis=1)

        import pickle
        pickle.dump(self.oof_preds, open("predicts", 'wb'))

            #test_preds += (np.array(model.predict(self.to_predict_X, verbose=1)[0])).ravel()
        #test_preds /= 5
        #self.preds = test_preds

        #self.save_result(test_preds)

        from sklearn.metrics import roc_auc_score
        # it can Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        #roc_auc_score(self.train_y>0.5, oof_preds)
        # print("Will start tf data handling")
        # train_y_aux = np.concatenate((np.expand_dims(train_y, axis=1), train_y_aux), axis=1)
        # global _d
        # global _dd # for debug

        # _d = train_y_aux
        # ds = tf.data.Dataset.from_tensor_slices(train_y_aux)
        # _dd = ds
        # filename = 'train_X_data.tfrecord'
        # writer = tf.data.experimental.TFRecordWriter(filename)
        # writer.write(ds)
        # print("done save tfrecord")

        # train_data = train_input_fn_general(train_X, train_y_aux, args.batch_size, True, split_id=fold, n_splits=5, handle_large=True, shuffle_size=1000, repeat=False)
        # val_data = eval_input_fn_general(train_X, train_y_aux, args.batch_size, True, split_id=fold, n_splits=5, handle_large=True) # for eval, no need to shuffle
        # print("after tf data handling")

    def save_result(self,predictions):
        submission = pd.DataFrame.from_dict({
            'id': self.emb.test_df_id,
            # 'id': test_df.id,
            'prediction': predictions
        })
        submission.to_csv('submission.csv', index=False)

    def prepare_identity_data(self):
        if self.train_y_identity is None:
            self.train_y_identity = self.emb.get_identity_columns_as_np()
        features = self.train_X
        labels = self.train_y_identity
        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        ds = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        iterator = ds.make_initializable_iterator()
        next_element = iterator.get_next()

        sess = K.get_session()
        itr = sess.run(iterator.initializer, feed_dict={
            features_placeholder: features,
            labels_placeholder: labels})
        #i = 0
        #while i < 10:
        #    try:
        #        el = sess.run(next_element)
        #        print("should be fine to enumerate")
        #        print(el)
        #        i += 1
        #    except tf.errors.OutOfRangeError:
        #        break
        return itr


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_steps', default=2, type=int,
                    help='number of training steps')
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='learing rate')

# _debug_train_data = None
def main(argv):
    args = parser.parse_args(argv[1:])


    global kernel
    kernel = KaggleKernel()

    print("Will start load data")
    kernel.emb = d.EmbeddingHandler()
    kernel.load_data()
    print("load data done")

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=30 * 60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=1,
        # save_checkpoints_steps = None
    )

    #model = kernel.build_model(len(kernel.train_y_aux[0]))
    #kernel.lstm_model_run()

    identity_classifier = tf.estimator.Estimator(
        model_fn=identity_model,
        params={
            'identity_out_num': len(d.IDENTITY_COLUMNS),
            'embedding_matrix': kernel.embedding_matrix,
            'learning_rate': args.learning_rate,
            'decay_steps': args.train_steps,
            #'optimizer': tf.compat.v1.train.AdamOptimizer(
            #learning_rate=tf.compat.v1.train.exponential_decay(
            #learning_rate=args.learning_rate,
            #global_step=tf.compat.v1.train.get_global_step(),
            #decay_steps=args.train_steps,
            #staircase=True,
            #decay_rate=0.5))
        },
        config=my_checkpointing_config,
        model_dir='/proc/driver/nvidia/identity-model') #warm_start_from="warm_model/model.ckpt-0")

    identity_classifier.train(
        input_fn=lambda: kernel.prepare_identity_data(),
        steps=args.train_steps) # steps is one as we repeat data in input_fn
    #kernel.lstm_model_run(args, my_checkpointing_config,
    #               params={
    #                   # Two hidden layers of 10 nodes each.
    #                   'hidden_units': [],
    #                   # The model must choose between 3 classes.
    #                   'n_classes': 120,
    #                   'batch_normalization': True,
    #                   'learning_rate': args.learning_rate,
    #                   'decay_steps': args.train_steps,
    #                   'train_steps': args.train_steps,
    #                   # 'optimizer': tf.compat.v1.train.AdamOptimizer(
    #                   # learning_rate=tf.compat.v1.train.exponential_decay(
    #                   # learning_rate=args.learning_rate,
    #                   # global_step=tf.compat.v1.train.get_global_step(),
    #                   # decay_steps=args.train_steps,
    #                   # staircase=True,
    #                   # decay_rate=0.5))



if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main([1])
    # tf.compat.v1.app.run(main)
    print("done")