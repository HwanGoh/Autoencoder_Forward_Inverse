import tensorflow as tf
import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                    Initialize Metrics and Storage Arrays                    #
###############################################################################
class Metrics:
    def __init__(self, dist_strategy):
        #=== Metrics ===#
        self.mean_loss_train = 0
        with dist_strategy.scope():
            self.mean_loss_train_VAE = tf.keras.metrics.Mean()
            self.mean_loss_train_KLD = tf.keras.metrics.Mean()
            self.mean_loss_train_post_mean = tf.keras.metrics.Mean()

            self.mean_loss_val = tf.keras.metrics.Mean()
            self.mean_loss_val_VAE = tf.keras.metrics.Mean()
            self.mean_loss_val_KLD = tf.keras.metrics.Mean()
            self.mean_loss_val_post_mean = tf.keras.metrics.Mean()

            self.mean_loss_test = tf.keras.metrics.Mean()
            self.mean_loss_test_VAE = tf.keras.metrics.Mean()
            self.mean_loss_test_KLD = tf.keras.metrics.Mean()
            self.mean_loss_test_post_mean = tf.keras.metrics.Mean()

            self.mean_relative_error_input_VAE = tf.keras.metrics.Mean()
            self.mean_relative_error_latent_encoder = tf.keras.metrics.Mean()
            self.mean_relative_error_input_decoder = tf.keras.metrics.Mean()

        #=== Initialize Metric Storage Arrays ===#
        self.storage_array_loss_train = np.array([])
        self.storage_array_loss_train_VAE = np.array([])
        self.storage_array_loss_train_KLD = np.array([])
        self.storage_array_loss_train_post_mean = np.array([])

        self.storage_array_loss_val = np.array([])
        self.storage_array_loss_val_VAE = np.array([])
        self.storage_array_loss_val_KLD = np.array([])
        self.storage_array_loss_val_post_mean = np.array([])

        self.storage_array_loss_test = np.array([])
        self.storage_array_loss_test_VAE = np.array([])
        self.storage_array_loss_test_KLD = np.array([])
        self.storage_array_loss_test_post_mean = np.array([])

        self.storage_array_relative_error_input_VAE = np.array([])
        self.storage_array_relative_error_latent_encoder = np.array([])
        self.storage_array_relative_error_input_decoder = np.array([])

###############################################################################
#                             Update Tensorboard                              #
###############################################################################
    def update_tensorboard(self, summary_writer, epoch):

        with summary_writer.as_default():
            tf.summary.scalar('loss_train',
                    self.mean_loss_train, step=epoch)
            tf.summary.scalar('loss_train_VAE',
                    self.mean_loss_train_VAE.result(), step=epoch)
            tf.summary.scalar('loss_train_KLD',
                    self.mean_loss_train_KLD.result(), step=epoch)
            tf.summary.scalar('loss_train_post_mean',
                    self.mean_loss_train_post_mean.result(), step=epoch)

            tf.summary.scalar('loss_val',
                    self.mean_loss_val.result(), step=epoch)
            tf.summary.scalar('loss_val_VAE',
                    self.mean_loss_val_VAE.result(), step=epoch)
            tf.summary.scalar('loss_val_KLD',
                    self.mean_loss_val_KLD.result(), step=epoch)
            tf.summary.scalar('loss_val_post_mean',
                    self.mean_loss_val_post_mean.result(), step=epoch)

            tf.summary.scalar('loss_test',
                    self.mean_loss_test.result(), step=epoch)
            tf.summary.scalar('loss_test_VAE',
                    self.mean_loss_test_VAE.result(), step=epoch)
            tf.summary.scalar('loss_test_KLD',
                    self.mean_loss_test_KLD.result(), step=epoch)
            tf.summary.scalar('loss_test_post_mean',
                    self.mean_loss_test_post_mean.result(), step=epoch)

            tf.summary.scalar('relative_error_input_VAE',
                    self.mean_relative_error_input_VAE.result(), step=epoch)
            tf.summary.scalar('relative_error_latent_encoder',
                    self.mean_relative_error_latent_encoder.result(), step=epoch)
            tf.summary.scalar('relative_error_input_decoder',
                    self.mean_relative_error_input_decoder.result(), step=epoch)

###############################################################################
#                            Update Storage Arrays                            #
###############################################################################
    def update_storage_arrays(self):
        self.storage_array_loss_train =\
                np.append(self.storage_array_loss_train,
                        self.mean_loss_train)
        self.storage_array_loss_train_VAE =\
                np.append(self.storage_array_loss_train_VAE,
                        self.mean_loss_train_VAE.result())
        self.storage_array_loss_train_KLD =\
                np.append(self.storage_array_loss_train_KLD,
                        self.mean_loss_train_KLD.result())
        self.storage_array_loss_train_post_mean =\
                np.append(self.storage_array_loss_train_post_mean,
                        self.mean_loss_train_post_mean.result())

        self.storage_array_loss_val =\
                np.append(self.storage_array_loss_val,
                        self.mean_loss_val.result())
        self.storage_array_loss_val_VAE =\
                np.append(self.storage_array_loss_val_VAE,
                        self.mean_loss_val_VAE.result())
        self.storage_array_loss_val_KLD =\
                np.append(self.storage_array_loss_val_KLD,
                        self.mean_loss_val_KLD.result())
        self.storage_array_loss_val_post_mean =\
                np.append(self.storage_array_loss_val_post_mean,
                        self.mean_loss_val_post_mean.result())

        self.storage_array_loss_test =\
                np.append(self.storage_array_loss_test,
                        self.mean_loss_test.result())
        self.storage_array_loss_test_VAE =\
                np.append(self.storage_array_loss_test_VAE,
                        self.mean_loss_test_VAE.result())
        self.storage_array_loss_test_KLD =\
                np.append(self.storage_array_loss_test_KLD,
                        self.mean_loss_test_KLD.result())
        self.storage_array_loss_test_post_mean =\
                np.append(self.storage_array_loss_test_post_mean,
                        self.mean_loss_test_post_mean.result())

        self.storage_array_relative_error_input_VAE =\
                np.append(self.storage_array_relative_error_input_VAE,
                        self.mean_relative_error_input_VAE.result())
        self.storage_array_relative_error_latent_encoder =\
                np.append(self.storage_array_relative_error_latent_encoder,
                        self.mean_relative_error_latent_encoder.result())
        self.storage_array_relative_error_input_decoder =\
                np.append(self.storage_array_relative_error_input_decoder,
                        self.mean_relative_error_input_decoder.result())
        self.storage_array_relative_gradient_norm =\
                np.append(self.storage_array_relative_gradient_norm,
                        self.relative_gradient_norm)

###############################################################################
#                                 Reset Metrics                               #
###############################################################################
    def reset_metrics(self):
        self.mean_loss_train.reset_states()
        self.mean_loss_train_VAE.reset_states()
        self.mean_loss_train_KLD.reset_states()
        self.mean_loss_train_post_mean.reset_states()

        self.mean_loss_val.reset_states()
        self.mean_loss_val_VAE.reset_states()
        self.mean_loss_val_KLD.reset_states()
        self.mean_loss_val_post_mean.reset_states()

        self.mean_loss_test.reset_states()
        self.mean_loss_test_VAE.reset_states()
        self.mean_loss_test_KLD.reset_states()
        self.mean_loss_test_post_mean.reset_states()

        self.mean_relative_error_input_VAE.reset_states()
        self.mean_relative_error_latent_encoder.reset_states()
        self.mean_relative_error_input_decoder.reset_states()

###############################################################################
#                                 Save Metrics                                #
###############################################################################
    def save_metrics(self, file_paths):
        metrics_dict = {}
        metrics_dict['loss_train'] = self.storage_array_loss_train
        metrics_dict['loss_train_VAE'] = self.storage_array_loss_train_VAE
        metrics_dict['loss_train_KLD'] = self.storage_array_loss_train_KLD
        metrics_dict['loss_train_post_mean'] = self.storage_array_loss_train_post_mean

        metrics_dict['loss_val'] = self.storage_array_loss_val
        metrics_dict['loss_val_VAE'] = self.storage_array_loss_val_VAE
        metrics_dict['loss_val_KLD'] = self.storage_array_loss_val_KLD
        metrics_dict['loss_val_post_mean'] = self.storage_array_loss_val_post_mean

        metrics_dict['relative_error_input_VAE'] =\
                self.storage_array_relative_error_input_VAE
        metrics_dict['relative_error_latent_encoder'] =\
                self.storage_array_relative_error_latent_encoder
        metrics_dict['relative_error_input_decoder'] =\
                self.storage_array_relative_error_input_decoder

        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(file_paths.NN_savefile_name + "_metrics" + '.csv', index=False)
