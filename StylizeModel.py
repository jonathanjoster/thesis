import os
import random
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import LeakyReLU
from LeakyAlt import LeakyAlt
from keras.layers import Input, Multiply, GlobalAveragePooling1D, Softmax, Add

_stage_label = ['W', 'N1', 'N2', 'N3', 'R']
five_stage_idx = [0, 187, 212, 137, 271]
_ylim1 = [-150., 150.]
_ylim3 = [0., 30.]
_width = 1/18
_alpha = .8
_fontsize = 30

def _fft(data):
    Fs = 120
    data = data.ravel()
    N = len(data)
    if N > 4000:
        print('Seems to be some error; got data length:', N)
        return
    window = np.hanning(N)

    freq = np.fft.fftfreq(N, 1.0/Fs)
    F = np.fft.fft(data * window)
    F = abs(F)/(N/2)
    F[0] = F[0]/2
    return freq[:N//2], F[:N//2] * 1/(sum(window)/2/N)

def print_fft(data, idx=29):
    """comparison between raw/reconstructed data"""    
    data = data[idx].ravel()
    
    x_fft, y_fft = _fft(data)

    _, ax = plt.subplots(1, 2, figsize=(24, 4))
    ax[0].plot(data)
    ax[0].set_title(f'Raw data')
    ax[1].plot(x_fft, y_fft, color='tab:orange')
    ax[1].set_title('FFT(Raw data)')
    plt.show()

class StylizeModel:
    def __init__(self, eeg, hyp, x_test, y_test) -> None:
        self.model = keras.models.load_model('./data/model_7746.h5', custom_objects={'LeakyReLU': LeakyReLU,
                                                                                     'LeakyAlt': LeakyAlt,
                                                                                     'root_mean_squared_error': StylizeModel.root_mean_squared_error,
                                                                                     'edge': 4})
        print('Model loaded')
        self.eeg = eeg
        self.hyp = hyp
        self.x_test = x_test
        self.y_test = y_test
        self.encoder, self.decoder = self._get_en_decoder()

        if self.eeg is not None:
            self.latent = self.encoder.predict(self.eeg)
        w, _ = self.decoder.get_layer(name='squeeze').get_weights()
        self.n_mfilter = w.shape[1] # w.shape: (kernel_size, f_in, f_out)
        self.edge = self.encoder.output_shape[-1] - self.n_mfilter
        self.imp_ascending = self._get_importance()

        self.rmse_transition = self.rec_transition = None

    def evaluate(self):
        """show stylize model's performance"""
        pred_test_oh, pred_test_wave = self.model.predict(self.x_test)
        pred_test_score = np.argmax(pred_test_oh, axis=-1)

        print(pd.DataFrame(confusion_matrix(self.y_test, pred_test_score), columns=_stage_label, index=_stage_label))
        print(pd.DataFrame(confusion_matrix(self.y_test, pred_test_score, normalize='true'), columns=_stage_label, index=_stage_label))
            
        print(classification_report(self.y_test, pred_test_score, target_names=_stage_label, digits=4, zero_division=0))

        print_fft(self.x_test)
        print_fft(pred_test_wave)
        
    def _get_en_decoder(self):
        StylizeModel.init_seed()
        kernel_size = 32
        inputs       = Input((3600, 1))
        x_1          = self.model.get_layer(name=f'ord_{kernel_size}_1')(inputs)
        x_2          = self.model.get_layer(name=f'alt_{kernel_size}_1')(inputs)
        x_1          = self.model.get_layer(name=f'ord_{kernel_size}_2')(x_1)
        x_2          = self.model.get_layer(name=f'alt_{kernel_size}_2')(x_2)
        outputs      = Add()([x_1, x_2])
        encoder = keras.Model(inputs, outputs)

        inputs       = Input(shape=encoder.output_shape[1:])
        mask_layer   = Input(shape=encoder.output_shape[1:])
        x            = Multiply()([inputs, mask_layer])
        enc          = self.model.get_layer(name=f'dec_{kernel_size}')(x)
        enc          = self.model.get_layer(name='rec')(enc)
        scoring      = self.model.get_layer(name='lambda')(x)
        scoring      = self.model.get_layer(name='ln')(scoring)
        scoring      = self.model.get_layer(name='squeeze')(scoring)
        scoring      = GlobalAveragePooling1D()(scoring)
        scoring      = Softmax(name='scoring')(scoring)
        decoder = keras.Model([inputs, mask_layer], [scoring, enc])
        
        return encoder, decoder
        
    def _get_importance(self):
        imp_ascending = [16, 23, 17, 2, 5, 7, 4, 25, 20, 21, 12, 15, 22, 26, 10, 24, 11, 27, 19, 9, 1, 8, 14, 18, 13, 3, 6, 0]
        return imp_ascending
            
    def _get_transition(self):
        self.rmse_transition = []
        self.rec_transition = []
        for i in tqdm.tqdm(range(len(self.imp_ascending)+1), desc='### calculate transition'):
            mask = np.ones_like(self.latent)
            for j in range(i):
                mask[:, :, self.edge//2+self.imp_ascending[j]] = 0
            pred_wave = self.decoder.predict([self.latent, mask])[1]
            
            rmse_val = StylizeModel.root_mean_squared_error(self.eeg, pred_wave)
            self.rmse_transition.append(rmse_val.numpy())
            
            rec = np.array([pred_wave[j] for j in five_stage_idx]) # (5, 3600, 1)
            self.rec_transition.append(rec)
        
    def show_rmse_transition(self):
        if self.rmse_transition is None:
            self._get_transition()

        plt.figure(figsize=(15, 3))
        plt.plot(self.rmse_transition)
        plt.xticks(range(len(self.rmse_transition)))
        plt.xlabel('Number of latent vector removed', fontsize=20)
        plt.ylabel('RMSE', fontsize=20)
        plt.grid()
        plt.show()
        
    def show_rec_transition(self, row=1):
        if self.rec_transition is None:
            self._get_transition()
            
        def _plot_some(ax, data, label, c=None):
            ax.plot(data, label=label, c=c)
            ax.set_ylim(_ylim1)
            x_fft, y_fft = _fft(data.ravel())
            ax3 = ax.twinx().twiny()
            ax3.plot(x_fft, y_fft, c='tab:orange', label=f'FFT({label})')
            ax3.set_ylim(_ylim3)
            h1, l1 = ax.get_legend_handles_labels()
            h3, l3 = ax3.get_legend_handles_labels()
            ax.legend(h1 + h3, l1 + l3)
            plt.tight_layout()

        _, ax = plt.subplots(1+row, 5, figsize=(30, 4*(1+row)), sharey='row')
        for i, idx in enumerate(five_stage_idx):
            ax1 = ax[0][i]
            _plot_some(ax1, self.x_test[idx], label='raw')
            ax1.set_title(f'Raw data, {_stage_label[i]}', fontsize=_fontsize)
        for i in tqdm.tqdm(range(row), desc='### plot rec transition '):
            for j in range(5):
                ax1 = ax[i+1][j]
                _plot_some(ax1, self.rec_transition[i][j], label='rec', c='navy') # specify threshold
                ax1.set_title(f'Removed {i}th({self.imp_ascending[i]}), {_stage_label[j]}', fontsize=_fontsize)
        plt.show()
        
    def compare(self, idx, extent):
        """comparison of raw/rec signals"""
        print(f'### Signal index: {idx}, stylized to {extent}/{self.n_mfilter}')
        
        mask = np.ones_like(self.latent)
        for j in range(extent):
            mask[:, :, self.edge//2+self.imp_ascending[j]] = 0
        stylized = self.decoder.predict([self.latent, mask])[1]

        _, ax = plt.subplots(2, 1, figsize=(20, 6), sharey='row')
        ax1 = ax[0]
        ax2 = ax1.twinx()
        ax3 = ax2.twiny()
        ax1.plot(self.x_test[idx], label='raw')
        x_fft, y_fft = _fft(self.x_test[idx].ravel())
        ax3.bar(x_fft, y_fft, color='tab:orange', width=_width, alpha=_alpha)

        ax1.set_xticks(np.arange(0, 3600+1, 600))
        ax1.set_ylim(_ylim1)
        ax3.set_xlabel('frequency[Hz]', fontsize=_fontsize)
        ax3.set_ylim(_ylim3)
        plt.tight_layout()
        ##################################
        ax1 = ax[1]
        ax2 = ax1.twinx()
        ax3 = ax2.twiny()
        ax1.plot(stylized[idx], c='navy')
        x_fft, y_fft = _fft(stylized[idx].ravel())
        ax3.bar(x_fft, y_fft, color='tab:orange', width=_width, alpha=_alpha)

        ax1.set_xticks(np.arange(0, 3600+1, 600))
        ax1.set_ylim(_ylim1)
        ax1.set_xlabel(f'Time[s]', fontsize=_fontsize)
        ax3.set_ylim(_ylim3)
        plt.tight_layout()
        plt.show()
        
    def show_ttp(self):
        """show TTP(tend to predict) and plot"""
        self.sensitive_to = [set(), set(), set(), set(), set()]
        self.ttp = {}

        for i, imp in tqdm.tqdm(enumerate(self.imp_ascending), desc='calculate ttp       ', total=self.n_mfilter):
            mask = np.zeros_like(self.latent)
            mask[:, :, self.edge//2+imp] = 1
            pred_oh = self.decoder.predict([self.latent, mask])[0]
            
            cm = confusion_matrix(self.hyp, np.argmax(pred_oh, axis=-1))
            pred_amb = np.argmax(np.sum(cm, axis=0))
            self.sensitive_to[pred_amb].add(i)
            self.ttp[i] = _stage_label[pred_amb]

        print('imp_ascending:', self.imp_ascending)

        for i in range(5):
            print(f'Sensitive to {_stage_label[i]}: {self.sensitive_to[i]}')

        print(self.ttp)
        
        imp_score = [0] * len(self.imp_ascending)
        for i, idx in enumerate(self.imp_ascending):
            imp_score[idx] = i / (len(self.imp_ascending)-1) * 100

        plt.figure(figsize=(20, 2))
        sns.heatmap([imp_score], cmap='Reds', annot=True, cbar=True, fmt='.1f')
        plt.title('Importance')
        plt.xlabel('index')
        plt.yticks([])
        plt.show()

        plt.figure(figsize=(10, 5))
        colorlist = ['r', 'b', 'c', 'm', 'g']
        for j in range(5):
            for i in self.sensitive_to[j]:
                plt.scatter(j, imp_score[i], color=colorlist[j])
                # plt.text(j+.05, imp_score[i], f'({i}){imp_score[i]:.1f}')
                plt.text(j+.05, imp_score[i], f'({i})')
        plt.xlabel('sensitive to')
        plt.xticks(range(5), _stage_label)
        plt.ylabel('Importance')
        plt.title('Importance / Tend to predict')
        plt.show()

    @classmethod
    def init_seed(cls, seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # session_conf = tf.compat.v1.ConfigProto(
        #     intra_op_parallelism_threads=1,
        #     inter_op_parallelism_threads=1
        # )
        # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        # tf.compat.v1.keras.backend.set_session(sess)

    @classmethod
    def root_mean_squared_error(cls, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 