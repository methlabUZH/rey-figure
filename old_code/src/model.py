from helpers import *

import tensorflow as tf
import numpy as np
import os
from sklearn import metrics
from scipy import stats
from random import randint

from config import DATA_AUGMENTATION, LOCAL, DEBUG, REGRESSOR_MODE, BINNING, LABEL_FORMAT, VAL_BINNING, \
    CLASSIFICATION_ENCODER, CONV_LAYERS, DROPOUT, TEST
from helpers import log_validation_predictions, write_validation_errors
from preprocess import augmented_CANVAS_SIZE, CANVAS_SIZE, one_hot_encoding, bin_numbers, \
    bin_numbers_continuous, BIN_LOCATIONS, postprocess_binned_labels, weighted_classes_encoding, \
    ordinal_classification_encoding, ordinal_classification_bin_number

if DATA_AUGMENTATION:
    CANVAS_SIZE = augmented_CANVAS_SIZE


class CNN:
    def __init__(self, run_name, fold, batch_size=128, dropout_ratio=DROPOUT):
        self.run_name = run_name
        self.fold = fold
        self.batch_size = batch_size
        self.dropout_ratio = dropout_ratio
        self.train_ids = None
        self.show_weight_summaries = False  # tensorboard summaries also for filters and biases
        self.build_model()

    def prepareX(self, X):
        return X

    def prepareY(self, y):
        # classifier
        if not REGRESSOR_MODE:
            # LABEL_FORMAT has no effect for classification -> always only take full score
            if LABEL_FORMAT == 'one-per-item':
                y = y[:, 18]
            elif LABEL_FORMAT == 'three-per-item':
                y = y[:, 54]
            if CLASSIFICATION_ENCODER == 'one-hot':
                y = one_hot_encoding(y)
            elif CLASSIFICATION_ENCODER == 'weighted':
                y = weighted_classes_encoding(y)
            elif CLASSIFICATION_ENCODER == 'ordinal':
                y = ordinal_classification_encoding(y)
            else:
                raise ValueError("invalid CLASSIFICATION_ENCODER")
            return y

        # regressor
        else:
            if LABEL_FORMAT == 'one':
                y = np.reshape(y, [np.shape(y)[0], 1])  # reshape to (?, 1)
            elif LABEL_FORMAT == 'one-per-item':
                y = np.reshape(y, [np.shape(y)[0], 19])  # reshape to (?, 19)
            elif LABEL_FORMAT == 'three-per-item':
                y = np.reshape(y, [np.shape(y)[0], 55])  # reshape to (?, 55)
            else:
                raise ValueError("invalid LABEL_FORMAT")

            if BINNING == 'discrete':
                if LABEL_FORMAT == 'one':
                    y = bin_numbers(y)
                elif LABEL_FORMAT == 'one-per-item':
                    y[:, 18] = bin_numbers(y[:, 18])
                elif LABEL_FORMAT == 'three-per-item':
                    y[:, 54] = bin_numbers(y[:, 54])
                else:
                    raise ValueError("invalid LABEL_FORMAT")

            elif BINNING == 'continuous':
                if LABEL_FORMAT == 'one':
                    y = bin_numbers_continuous(y)
                elif LABEL_FORMAT == 'one-per-item':
                    y[:, 18] = bin_numbers_continuous(y[:, 18])
                elif LABEL_FORMAT == 'three-per-item':
                    y[:, 54] = bin_numbers_continuous(y[:, 54])
                else:
                    raise ValueError("invalid LABEL_FORMAT")

            else:
                if BINNING != "none": raise ValueError("invalid BINNING")

        return y

    def prepare_data(self, X, y=None):
        # binning_mode is either 'discrete' or 'continuous'
        X_prep = self.prepareX(X)
        if (y is None):
            return X_prep
        y_prep = self.prepareY(y)
        return X_prep, y_prep

    def build_main_model(self):
        input_size = tf.shape(self.input)[0]

        input = tf.expand_dims(self.input, 3)

        with tf.variable_scope('conv1') as scope:
            filters = tf.get_variable("filter", [5, 5, 1, 32], tf.float32)

            conv = tf.nn.conv2d(input, filters, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [32], tf.float32)
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.leaky_relu(pre_activation, name=scope.name)

        if self.show_weight_summaries:
            tf.summary.histogram('conv1_filters', filters)
            tf.summary.histogram('conv1_biases', biases)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            filters = tf.get_variable("filter", [5, 5, 32, 32], tf.float32)
            conv = tf.nn.conv2d(pool1, filters, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [32], tf.float32)
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.leaky_relu(pre_activation, name=scope.name)

        if self.show_weight_summaries:
            tf.summary.histogram('conv2_filters', filters)
            tf.summary.histogram('conv2_biases', biases)

        # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        if CONV_LAYERS == 2:

            reshaped = tf.reshape(pool2, [input_size, 29 * 38 * 32])

        elif CONV_LAYERS > 2:

            # conv3
            with tf.variable_scope('conv3') as scope:
                filters = tf.get_variable("filter", [5, 5, 32, 64], tf.float32)
                conv = tf.nn.conv2d(pool2, filters, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', [64], tf.float32)
                pre_activation = tf.nn.bias_add(conv, biases)
                conv3 = tf.nn.leaky_relu(pre_activation, name=scope.name)

            if self.show_weight_summaries:
                tf.summary.histogram('conv3_filters', filters)
                tf.summary.histogram('conv3_biases', biases)

            # pool3
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        if CONV_LAYERS == 3:
            reshaped = tf.reshape(pool3, [input_size, 15 * 19 * 64])

        elif CONV_LAYERS == 4:
            # conv4
            with tf.variable_scope('conv4') as scope:
                filters = tf.get_variable("filter", [5, 5, 64, 64], tf.float32)
                conv = tf.nn.conv2d(pool3, filters, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', [64], tf.float32)
                pre_activation = tf.nn.bias_add(conv, biases)
                conv4 = tf.nn.leaky_relu(pre_activation, name=scope.name)

            if self.show_weight_summaries:
                tf.summary.histogram('conv4_filters', filters)
                tf.summary.histogram('conv4_biases', biases)

            # pool3
            pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            reshaped = tf.reshape(pool4, [input_size, 8 * 10 * 64])

        dense = tf.layers.dense(reshaped, 1024, activation=tf.nn.leaky_relu, name="dense",
                                bias_initializer=tf.glorot_uniform_initializer())

        self.intermediate = dense

        tf.summary.histogram('dense', dense)

        dense_dropped = tf.layers.dropout(dense, rate=self.dropout_ratio, training=self.is_training)

        if REGRESSOR_MODE:
            if LABEL_FORMAT == 'one-per-item':
                prediction = tf.layers.dense(dense_dropped, 19, activation=tf.nn.leaky_relu, name="prediction")
            elif LABEL_FORMAT == 'three-per-item':
                prediction = tf.layers.dense(dense_dropped, 55, activation=tf.nn.leaky_relu, name="prediction")
            else:
                prediction = tf.layers.dense(dense_dropped, 1, activation=tf.nn.leaky_relu, name="prediction")
            loss = tf.losses.mean_squared_error(tf.cast(self.labels, tf.float32), prediction)
            prediction_bin = None
            accuracy = None

        else:
            if CLASSIFICATION_ENCODER == 'ordinal':
                prediction = tf.layers.dense(dense_dropped, len(BIN_LOCATIONS), activation=tf.nn.sigmoid,
                                             name="prediction_ordinal")

                # produce bin numbers (like ordinal_classification_bin_number() but in tensorflow)
                weighting = tf.range(tf.shape(prediction)[1])
                weighted_labels = tf.multiply(tf.cast(tf.greater(prediction, 0.5), tf.int32), weighting)
                prediction_bin = tf.argmax(weighted_labels, axis=1)
                weighting = tf.range(tf.shape(self.labels)[1])
                weighted_labels = tf.multiply(tf.cast(tf.greater(self.labels, 0.5), tf.int32), weighting)
                labels_bin = tf.argmax(weighted_labels, axis=1)

                loss = tf.losses.mean_squared_error(self.labels, prediction)
                accuracy, _ = tf.metrics.accuracy(labels_bin, prediction_bin)
            else:
                prediction = tf.layers.dense(dense_dropped, len(BIN_LOCATIONS), activation=tf.nn.leaky_relu,
                                             name="prediction")
                prediction_bin = tf.argmax(prediction, axis=1)
                labels_bin = tf.argmax(self.labels, axis=1)
                loss = tf.losses.softmax_cross_entropy(self.labels, prediction)
                accuracy, _ = tf.metrics.accuracy(labels_bin, prediction_bin)

        # collect summaries in main model
        main_model_summaries = tf.summary.merge_all()

        # saver to save model parameters
        self.saver = tf.train.Saver()

        return loss, accuracy, prediction, prediction_bin, main_model_summaries

    def build_model(self):
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float32, [None, CANVAS_SIZE[0], CANVAS_SIZE[1]], name="input")
        if REGRESSOR_MODE:
            if LABEL_FORMAT == 'one-per-item':
                self.labels = tf.placeholder(tf.float32, [None, 19], name="labels")
            elif LABEL_FORMAT == 'three-per-item':
                self.labels = tf.placeholder(tf.float32, [None, 55], name="labels")
            else:
                self.labels = tf.placeholder(tf.float32, [None, 1], name="labels")
        else:
            self.labels = tf.placeholder(tf.float32, [None, len(BIN_LOCATIONS)], name="labels")
        self.is_training = tf.placeholder(tf.bool)

        loss, accuracy, self.prediction, self.prediction_bin, main_model_summaries = self.build_main_model()

        # TODO add correct elements to Tensorboard summary
        # summaries for tensorboard
        summary_loss = tf.summary.scalar('loss', loss)
        if REGRESSOR_MODE:
            if LABEL_FORMAT == 'one-per-item':
                summary_prediction = tf.summary.histogram('prediction', self.prediction[:, 18])
                mean, var = tf.nn.moments(self.prediction[:, 18], axes=[0], name="mean_var")
            elif LABEL_FORMAT == 'three-per-item':
                summary_prediction = tf.summary.histogram('prediction', self.prediction[:, 54])
                mean, var = tf.nn.moments(self.prediction[:, 54], axes=[0], name="mean_var")
            else:
                summary_prediction = tf.summary.histogram('prediction', self.prediction)
                mean, var = tf.nn.moments(self.prediction, axes=[0], name="mean_var")
        else:
            summary_prediction = tf.summary.histogram('prediction_bin', self.prediction_bin)
            mean, var = tf.nn.moments(self.prediction_bin, axes=[0], name="mean_var")
            summary_accuracy = tf.summary.scalar('accuracy', accuracy)
        summary_pred_mean = tf.summary.scalar('prediction_mean', tf.squeeze(mean))
        summary_pred_var = tf.summary.scalar('prediction_variance', tf.squeeze(var))

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(.001, self.global_step,
                                                   10000, 0.1, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer")
        self.training_step = optimizer.minimize(loss=loss, global_step=self.global_step)

        if not LOCAL:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._session = tf.Session(config=config)
        else:
            self._session = tf.Session()

        self._session.run(tf.global_variables_initializer())
        self._session.run(tf.local_variables_initializer())

        # merge all summaries for tensorboard
        self.all_summaries = tf.summary.merge_all()
        # merge all summaries except loss (for prediction, when label isn't given)
        # add summary accuracy again
        if REGRESSOR_MODE:
            self.all_summaries_predict = tf.summary.merge([summary_prediction, summary_pred_mean, summary_pred_var])
        else:
            self.all_summaries_predict = tf.summary.merge(
                [summary_prediction, summary_accuracy, summary_pred_mean, summary_pred_var])

        self._session.graph.finalize()

        # writer for tensorboard visualization
        self.train_writer = tf.summary.FileWriter(os.path.join('../summaries', self.run_name, str(self.fold), "train"),
                                                  self._session.graph)
        self.validation_writer = tf.summary.FileWriter(
            os.path.join('../summaries', self.run_name, str(self.fold), "val"))

    def predict(self, X, summary_writer=False, summary_step=False):
        X = np.copy(X)
        X = self.prepare_data(X)

        results = []

        ids = BatchIds(X.shape[0], random=False, reset=False)
        first_batch = True
        while not ids.empty():
            curr_batch_ids = ids.get_batch_ids(self.batch_size)
            curr_batch = X[curr_batch_ids]

            if DATA_AUGMENTATION and not TEST:
                # use if data augmentation
                int_batch = np.empty([curr_batch.shape[0], 116, 150])
                for i in range(curr_batch.shape[0]):
                    int_batch[i, :, :] = curr_batch[i, 0, :, :]
                curr_batch = int_batch

            if summary_writer and summary_step and first_batch:
                # write tensorboard for the first batch
                summary, curr_result = self._session.run([self.all_summaries_predict, self.prediction],
                                                         {self.input: curr_batch, self.is_training: False})
                summary_writer.add_summary(summary, summary_step)
                first_batch = False
            else:
                curr_result = self._session.run(self.prediction,
                                                {self.input: curr_batch, self.is_training: False})

            results.append(curr_result)

        results = np.concatenate(results, axis=0)
        return results

    def fit(self, X, y, trainingsteps=3, globalstep=0, reset_ids=False):
        y = np.copy(y)
        X = np.copy(X)
        X, y = self.prepare_data(X, y)

        if (reset_ids or (self.train_ids is None)):
            self.train_ids = BatchIds(X.shape[0])
        for i in range(trainingsteps):
            curr_batch_ids = self.train_ids.get_batch_ids(self.batch_size)
            curr_data = X[curr_batch_ids]
            curr_labels = y[curr_batch_ids]

            if DATA_AUGMENTATION:
                # data augmentation
                int_batch = np.empty([curr_data.shape[0], 116, 150])
                for i in range(curr_data.shape[0]):
                    int_batch[i, :, :] = curr_data[i, randint(0, 9), :, :]
                curr_data = int_batch

            summary, _ = self._session.run([self.all_summaries, self.training_step],
                                           {
                                               self.input: curr_data,
                                               self.labels: curr_labels,
                                               self.is_training: True
                                           })
            self.train_writer.add_summary(summary, globalstep * trainingsteps + i)

    def get_intermediate(self, X):
        X = np.copy(X)
        X = self.prepare_data(X)

        intermediates = []

        curr_intermediate = self._session.run(self.intermediate,
                                              {self.input: X, self.is_training: False})

        intermediates.append(curr_intermediate)

        intermediates = np.concatenate(intermediates, axis=0)

        return intermediates

    def validate(self, X, y, summary_writer=None, summary_step=None, files=None, log_results=False,
                 log_results_filename=None):
        y = np.copy(y)
        X = np.copy(X)
        y = self.prepareY(y)
        prediction = self.predict(X, summary_writer=summary_writer, summary_step=summary_step)
        extra_information = np.copy(prediction)  # not used for validation, but for logging results

        if REGRESSOR_MODE:
            if LABEL_FORMAT == 'one-per-item':
                prediction = prediction[:, 18]
                y = y[:, 18]
            elif LABEL_FORMAT == 'three-per-item':
                prediction = prediction[:, 54]
                y = y[:, 54]
            if BINNING != 'none':
                # convert to integer bins to ensure fair comparison between methods
                y = postprocess_binned_labels(y)
                prediction = postprocess_binned_labels(prediction)
            if BINNING == 'none' and VAL_BINNING:
                # trained on original labels, but validation should be on bins
                y = bin_numbers(y)
                prediction = bin_numbers(prediction)
            mse = metrics.mean_squared_error(y, prediction)
        else:
            if CLASSIFICATION_ENCODER == 'ordinal':
                prediction = ordinal_classification_bin_number(prediction)
                y = ordinal_classification_bin_number(y)
            else:
                prediction = np.argmax(prediction, axis=1)
                y = np.argmax(y, axis=1)
            mse = metrics.mean_squared_error(y, prediction)

        if DEBUG:
            print("Validation prediction statistic")
            print(stats.describe(prediction))

        if (log_results):
            log_validation_predictions(y, prediction, files, log_results_filename, extra_information)
            write_validation_errors(y, prediction, files, self.run_name)

        if summary_writer and summary_step:
            summary_val_loss = tf.Summary(value=[
                tf.Summary.Value(tag="loss", simple_value=mse),
            ])
            summary_writer.add_summary(summary_val_loss, summary_step)

        return mse

    def save_model(self):
        """ saves trained model so it can later be restored """
        model_path = os.path.join('../models', self.run_name, "model_fold" + str(self.fold) + ".ckpt")
        save_path = self.saver.save(self._session, model_path)
        print("Model saved in path: %s" % save_path)

    def restore_model(self, model_path):
        self.saver.restore(self._session, model_path)
        print("Model restored from path: %s" % model_path)
