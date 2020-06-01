import tensorflow as tf
import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                    Initialize Metrics and Storage Arrays                    #
###############################################################################
class Metrics:
    def __init__(self):
        #=== Metrics ===#
        self.mean_loss_train = tf.keras.metrics.Mean()
        self.mean_loss_train_autoencoder = tf.keras.metrics.Mean()
        self.mean_loss_train_encoder = tf.keras.metrics.Mean()
        self.mean_loss_train_decoder = tf.keras.metrics.Mean()

        self.mean_loss_val = tf.keras.metrics.Mean()
        self.mean_loss_val_autoencoder = tf.keras.metrics.Mean()
        self.mean_loss_val_encoder = tf.keras.metrics.Mean()
        self.mean_loss_val_decoder = tf.keras.metrics.Mean()

        self.mean_loss_test = tf.keras.metrics.Mean()
        self.mean_loss_test_autoencoder = tf.keras.metrics.Mean()
        self.mean_loss_test_encoder = tf.keras.metrics.Mean()
        self.mean_loss_test_decoder = tf.keras.metrics.Mean()

        self.mean_relative_error_data_autoencoder = tf.keras.metrics.Mean()
        self.mean_relative_error_latent_encoder = tf.keras.metrics.Mean()
        self.mean_relative_error_data_decoder = tf.keras.metrics.Mean()
        self.relative_gradient_norm = 0;

        #=== Initialize Metric Storage Arrays ===#
        self.storage_array_loss_train = np.array([])
        self.storage_array_loss_train_autoencoder = np.array([])
        self.storage_array_loss_train_encoder = np.array([])
        self.storage_array_loss_train_decoder = np.array([])

        self.storage_array_loss_val = np.array([])
        self.storage_array_loss_val_autoencoder = np.array([])
        self.storage_array_loss_val_encoder = np.array([])
        self.storage_array_loss_val_decoder = np.array([])

        self.storage_array_loss_test = np.array([])
        self.storage_array_loss_test_autoencoder = np.array([])
        self.storage_array_loss_test_encoder = np.array([])
        self.storage_array_loss_test_decoder = np.array([])

        self.storage_array_relative_error_data_autoencoder = np.array([])
        self.storage_array_relative_error_latent_encoder = np.array([])
        self.storage_array_relative_error_data_decoder = np.array([])

        self.storage_array_relative_gradient_norm = np.array([])

###############################################################################
#                             Update Tensorboard                              #
###############################################################################
    def update_tensorboard(self, summary_writer, epoch):

        with summary_writer.as_default():
            tf.summary.scalar('loss_training',
                    self.mean_loss_train.result(), step=epoch)
            tf.summary.scalar('loss_training_autoencoder',
                    self.mean_loss_train_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_training_encoder',
                    self.mean_loss_train_encoder.result(), step=epoch)
            tf.summary.scalar('loss_training_decoder',
                    self.mean_loss_train_decoder.result(), step=epoch)
            tf.summary.scalar('loss_val',
                    self.mean_loss_val.result(), step=epoch)
            tf.summary.scalar('loss_val_autoencoder',
                    self.mean_loss_val_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_val_encoder',
                    self.mean_loss_val_encoder.result(), step=epoch)
            tf.summary.scalar('loss_val_decoder',
                    self.mean_loss_val_decoder.result(), step=epoch)
            tf.summary.scalar('loss_test',
                    self.mean_loss_test.result(), step=epoch)
            tf.summary.scalar('loss_test_autoencoder',
                    self.mean_loss_test_autoencoder.result(), step=epoch)
            tf.summary.scalar('loss_test_encoder',
                    self.mean_loss_test_encoder.result(), step=epoch)
            tf.summary.scalar('loss_test_decoder',
                    self.mean_loss_test_decoder.result(), step=epoch)
            tf.summary.scalar('relative_error_data_autoencoder',
                    self.mean_relative_error_data_autoencoder.result(), step=epoch)
            tf.summary.scalar('relative_error_latent_encoder',
                    self.mean_relative_error_latent_encoder.result(), step=epoch)
            tf.summary.scalar('relative_error_data_decoder',
                    self.mean_relative_error_data_decoder.result(), step=epoch)
            tf.summary.scalar('relative_gradient_norm', self.relative_gradient_norm, step=epoch)

###############################################################################
#                            Update Storage Arrays                            #
###############################################################################
    def update_storage_arrays(self):
        self.storage_array_loss_train =\
                np.append(self.storage_array_loss_train,
                        self.mean_loss_train.result())
        self.storage_array_loss_train_autoencoder =\
                np.append(self.storage_array_loss_train_autoencoder,
                        self.mean_loss_train_autoencoder.result())
        self.storage_array_loss_train_encoder =\
                np.append(self.storage_array_loss_train_encoder,
                        self.mean_loss_train_encoder.result())
        self.storage_array_loss_train_decoder =\
                np.append(self.storage_array_loss_train_decoder,
                        self.mean_loss_train_decoder.result())
        self.storage_array_loss_val =\
                np.append(self.storage_array_loss_val,
                        self.mean_loss_val.result())
        self.storage_array_loss_val_autoencoder =\
                np.append(self.storage_array_loss_val_autoencoder,
                        self.mean_loss_val_autoencoder.result())
        self.storage_array_loss_val_encoder =\
                np.append(self.storage_array_loss_val_encoder,
                        self.mean_loss_val_encoder.result())
        self.storage_array_loss_val_decoder =\
                np.append(self.storage_array_loss_val_decoder,
                        self.mean_loss_val_decoder.result())
        self.storage_array_loss_test =\
                np.append(self.storage_array_loss_test,
                        self.mean_loss_test.result())
        self.storage_array_loss_test_autoencoder =\
                np.append(self.storage_array_loss_test_autoencoder,
                        self.mean_loss_test_autoencoder.result())
        self.storage_array_loss_test_encoder =\
                np.append(self.storage_array_loss_test_encoder,
                        self.mean_loss_test_encoder.result())
        self.storage_array_loss_test_decoder =\
                np.append(self.storage_array_loss_test_decoder,
                        self.mean_loss_test_decoder.result())
        self.storage_array_relative_error_data_autoencoder =\
                np.append(self.storage_array_relative_error_data_autoencoder,
                        self.mean_relative_error_data_autoencoder.result())
        self.storage_array_relative_error_latent_encoder =\
                np.append(self.storage_array_relative_error_latent_encoder,
                        self.mean_relative_error_latent_encoder.result())
        self.storage_array_relative_error_data_decoder =\
                np.append(self.storage_array_relative_error_data_decoder,
                        self.mean_relative_error_data_decoder.result())
        self.storage_array_relative_gradient_norm =\
                np.append(self.storage_array_relative_gradient_norm,
                        self.relative_gradient_norm)

###############################################################################
#                                 Reset Metrics                               #
###############################################################################
    def reset_metrics(self):
        self.mean_loss_train.reset_states()
        self.mean_loss_train_autoencoder.reset_states()
        self.mean_loss_train_encoder.reset_states()
        self.mean_loss_train_decoder.reset_states()
        self.mean_loss_val.reset_states()
        self.mean_loss_val_autoencoder.reset_states()
        self.mean_loss_val_encoder.reset_states()
        self.mean_loss_val_decoder.reset_states()
        self.mean_loss_test.reset_states()
        self.mean_loss_test_autoencoder.reset_states()
        self.mean_loss_test_encoder.reset_states()
        self.mean_loss_test_decoder.reset_states()
        self.mean_relative_error_data_autoencoder.reset_states()
        self.mean_relative_error_latent_encoder.reset_states()
        self.mean_relative_error_data_decoder.reset_states()

###############################################################################
#                                 Save Metrics                                #
###############################################################################
    def save_metrics(self, file_paths):
        metrics_dict = {}
        metrics_dict['loss_train'] = self.storage_array_loss_train
        metrics_dict['loss_train_autoencoder'] = self.storage_array_loss_train_autoencoder
        metrics_dict['loss_train_encoder'] = self.storage_array_loss_train_encoder
        metrics_dict['loss_train_decoder'] = self.storage_array_loss_train_decoder
        metrics_dict['loss_val'] = self.storage_array_loss_val
        metrics_dict['loss_val_autoencoder'] = self.storage_array_loss_val_autoencoder
        metrics_dict['loss_val_encoder'] = self.storage_array_loss_val_encoder
        metrics_dict['loss_val_decoder'] = self.storage_array_loss_val_decoder
        metrics_dict['relative_error_data_autoencoder'] =\
                self.storage_array_relative_error_data_autoencoder
        metrics_dict['relative_error_latent_encoder'] =\
                self.storage_array_relative_error_latent_encoder
        metrics_dict['relative_error_data_decoder'] = self.storage_array_relative_error_data_decoder
        metrics_dict['relative_gradient_norm'] = self.storage_array_relative_gradient_norm
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(file_paths.NN_savefile_name + "_metrics" + '.csv', index=False)
