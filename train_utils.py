"""
Defines a functions for training a Neural Network
"""

import _pickle as pickle
import os
import sys
import traceback

from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, TerminateOnNaN
from keras.layers import (Input, Lambda)
from keras.models import Model
from keras.optimizers import SGD

from data_generator import AudioGenerator
from models import ModelBuilder


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths],
        outputs=loss_out)
    return model


import matplotlib.pyplot as plt

import numpy as np
from IPython import display


class MetricsLogger(Callback):
    """Callback that accumulates epoch averages of metrics.

    This callback is automatically applied to every Keras model.

    # Arguments
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is in `on_epoch_end`.
            All others will be averaged in `on_epoch_end`.
    """

    def __init__(self, model_name, n_epochs=None, loss_limit=None):
        super().__init__()
        self.model_name = model_name
        self.n_epochs = n_epochs
        self.loss_limit = loss_limit

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        self.epoch = 0
        self.epochs = []
        self.losses = []
        self.val_losses = []

        self.fig, self.ax1 = plt.subplots()

        self.ax1.set(title=self.model_name)
        if self.n_epochs:
            self.ax1.set_xlim(0, self.n_epochs + 1)
        if self.loss_limit:
            self.ax1.set_ylim(0, self.loss_limit)
            plt.yticks(np.arange(0, self.loss_limit, 1 if self.loss_limit < 10 else self.loss_limit // 10))
        self.color1 = 'tab:green'
        self.ax1.set_ylabel('Loss')
        # self.ax1.tick_params(axis='y')

        # self.ax1.set_ylabel('Loss', color=self.color1)

        self.ax1.set_xlabel('Epochs')

        self.color2 = 'tab:red'
        # self.ax2 = self.ax1.twinx()  # instantiate a second axes that shares the same x-axis
        # self.ax2.set_ylabel("Validation loss", color=self.color2, labelpad=15)  # we already handled the x-label with ax1

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.epoch += 1
            self.epochs = np.append(self.epochs, self.epoch)
            self.losses = np.append(self.losses, logs['loss'])
            self.val_losses = np.append(self.val_losses, logs['val_loss'])

            plt.gca().cla()
            if self.n_epochs:
                self.ax1.set_xlim(0, self.n_epochs + 1)
            if self.loss_limit:
                self.ax1.set_ylim(0, self.loss_limit)
            self.ax1.set_xlabel('Epochs')

            line1 = self.ax1.plot(self.epochs, self.losses, color=self.color1, label="Loss")
            # self.ax1.tick_params(axis='y', labelcolor=self.color1)

            if self.n_epochs:
                step = 1
                while self.n_epochs // step > 10:
                    if step == 1:
                        step = 2
                    elif step == 2:
                        step = 5
                    else:
                        step += 5
                plt.xticks(np.arange(0, self.n_epochs + 1, step))

            line2 = self.ax1.plot(self.epochs, self.val_losses, color=self.color2, label="Validation loss")

            # self.ax2.set_ylim(0, 800)
            #
            # self.ax2.tick_params(axis='y', labelcolor=self.color2)
            # self.ax2.set_ylabel("Validation loss", color=self.color2, labelpad=15)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            self.ax1.grid()
            self.ax1.legend(lines, labels, loc="best")
            self.ax1.set(title=self.model_name)

            display.clear_output(wait=True)
            display.display(plt.gcf())
            self.fig.savefig(os.path.join("results", self.model_name + ".png"))
            history = dict()
            history["loss"] = self.losses
            history["val_loss"] = self.val_losses
            history["name"] = self.model_name

            # save model loss
            with open('results/' + self.model_name + ".pickle", 'wb') as f:
                pickle.dump(history, f)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        display.clear_output(wait=True)

        # self.model.summary()


def train_model(audio_gen: AudioGenerator,
                model_builder: ModelBuilder,
                # pickle_path,
                # save_model_path,
                # train_json='train_corpus.json',
                # valid_json='valid_corpus.json',
                optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                # optimizer=Adam(lr=1e-01),
                epochs=30,
                verbose=0,
                # sort_by_duration=False,
                loss_limit=400):
    # create a class instance for obtaining batches of data
    input_dim = audio_gen.input_dim
    if audio_gen.max_length is None:
        model = model_builder.model(input_shape=(None, input_dim), output_dim=29)
    else:
        model = model_builder.model(input_shape=(audio_gen.max_length, input_dim), output_dim=29)
    model_name = ("Spec" if audio_gen.spectrogram else "MFCC") + " " + model.name
    model.name = model_name
    save_model_path = model.name + ".h5"

    # add the training data to the generator
    # audio_gen.load_train_data(train_json)
    # audio_gen.load_validation_data(valid_json)
    # calculate steps_per_epoch
    num_train_examples = len(audio_gen.train_audio_paths)
    steps_per_epoch = num_train_examples // audio_gen.minibatch_size
    # calculate validation_steps
    num_valid_samples = len(audio_gen.valid_audio_paths)
    validation_steps = num_valid_samples // audio_gen.minibatch_size

    # add CTC loss to the NN specified in input_to_softmax
    pre_model = model
    model = add_ctc_loss(model)

    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    # make results/ directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')

    # add model_checkpoint
    model_checkpoint = ModelCheckpoint(filepath='results/' + save_model_path, verbose=0, save_best_only=True)
    terminate_on_na_n = TerminateOnNaN()
    if verbose > 0:
        callbacks = [model_checkpoint, terminate_on_na_n]
    else:
        metrics_logger = MetricsLogger(model_name=model_name, n_epochs=epochs, loss_limit=loss_limit)
        callbacks = [model_checkpoint, metrics_logger]
        # callbacks = [model_checkpoint, metrics_logger, terminate_on_na_n]

    try:
        # hist = \
        model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=steps_per_epoch,
                            epochs=epochs, validation_data=audio_gen.next_valid(),
                            validation_steps=validation_steps,
                            callbacks=callbacks, verbose=verbose)
        # hist.history["name"] = model_name
        # save model loss
        # pickle_file_name = 'results/' + pickle_path
        # print("Writing hist.history[\"name\"] = ", model_name, "to ", pickle_file_name)
        # with open(pickle_file_name, 'wb') as f:
        #     pickle.dump(hist.history, f)
    except KeyboardInterrupt:
        display.clear_output(wait=True)
        # print("Training interrupted")
    except Exception:
        try:
            exc_info = sys.exc_info()
        finally:
            # Display the *original* exception
            traceback.print_exception(*exc_info)
            del exc_info
    finally:
        pre_model.summary()
        del pre_model
        del model
    return model_name


from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text

def load_model(data_gen: AudioGenerator, model_builder: ModelBuilder):
    model = model_builder.model(input_shape=(None, data_gen.input_dim), output_dim=29)
    model.load_weights('results/' + ("Spec " if data_gen.spectrogram else "MFCC ") + model.name + '.h5')
    return model


def get_predictions(data_gen: AudioGenerator,
                    model,
                    partition, index, omit_true=False, print_line=True):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        model (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    # data_gen = AudioGenerator()
    # data_gen.load_train_data()
    # data_gen.load_validation_data()

    # obtain the true transcription and the audio features
    if data_gen is None:
        print("Data Generator is None!")
    if partition == 'validation':
        transcription = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcription = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    # obtain and decode the acoustic model's predictions
    prediction = model.predict(np.expand_dims(data_point, axis=0))
    output_length = [model.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
        prediction, output_length)[0][0]) + 1).flatten().tolist()

    # play the audio file, and display the true and predicted transcriptions
    # Audio(audio_path)
    input_type = "SPEC" if data_gen.spectrogram else "MFCC"
    if not omit_true:
        print('TRUE:      ' + transcription)
    print('PRED ' + input_type + ': ' + ''.join(int_sequence_to_text(pred_ints)))
    if print_line:
        print('-' * 82)
    return audio_path




from glob import glob


def plot_comparison(pickles=None, min_epoch=1, max_epoch=None, min_loss=90, max_loss=120):
    # obtain the paths for the saved model history
    # extract the name and loss history for each model

    if not pickles:
        pickles = sorted(glob("results/*.pickle"))
    model_names = [pickle.load(open(i, "rb"))['name'] for i in pickles]
    valid_loss = [pickle.load(open(i, "rb"))['val_loss'] for i in pickles]
    # train_loss = [pickle.load(open(i, "rb"))['loss'] for i in pickles]
    # save the number of epochs used to train each model
    num_epochs = [len(valid_loss[i]) for i in range(len(valid_loss))]

    fig = plt.figure(figsize=(16, 5))

    # plot the training loss vs. epoch for each model
    # ax1 = fig.add_subplot(121)
    # for i in range(len(pickles)):
    #     ax1.plot(np.linspace(1, num_epochs[i], num_epochs[i]),
    #              train_loss[i], label=model_names[i])
    # # clean up the plot
    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax1.set_xlim([1, max(num_epochs)])
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')

    # plot the validation loss vs. epoch for each model
    ax2 = fig.add_subplot(122)
    for i in range(len(pickles)):
        ax2.plot(range(min_epoch, num_epochs[i] + 1),
                 valid_loss[i], label=model_names[i])
    # clean up the plot
    ax2.grid()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlim([min_epoch, max_epoch if max_epoch is not None else max(num_epochs)])
    ax2.set_ylim([min_loss, max_loss])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.show()


