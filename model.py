# General collecting part of all the models

from . import model_segnet
from . import model_danet
from .keras_unet.models.custom_unet import custom_unet
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.utils import to_categorical
import numpy as np
from .common import *
import gzip
import dill
import os, shutil
from datetime import datetime
import json
from .keras_unet_collection.models import *
from .keras_unet_collection.losses import *
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from contextlib import redirect_stdout
from timeit import default_timer as timer

# from ._model_unet_2d import unet_2d
# from ._model_vnet_2d import vnet_2d
# from ._model_unet_plus_2d import unet_plus_2d
# from ._model_r2_unet_2d import r2_unet_2d
# from ._model_att_unet_2d import att_unet_2d
# from ._model_resunet_a_2d import resunet_a_2d
# from ._model_u2net_2d import u2net_2d
# from ._model_unet_3plus_2d import unet_3plus_2d
# from ._model_transunet_2d import transunet_2d
# from ._model_swin_unet_2d import swin_unet_2d

class TimingCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()

class SSegModel:
    def __init__(self,dataset_name,model_name,model_params,train_params):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_params = model_params
        self.train_params = train_params

        if 'loss' not in self.train_params:
            self.train_params['loss'] = 'cce'

        if(model_name == 'unet_2'):
            self.model = custom_unet(**model_params)
        elif(model_name == 'segnet'):
            self.model = model_segnet.segnet(**model_params)
        elif(model_name == 'danet'):
            self.model = model_danet.danet_resnet101(**model_params)
        elif(model_name == 'unet'):
            self.model = unet_2d(**model_params)
        elif(model_name == 'unet_crf'):
            self.model = unet_unet_2d_crf(**model_params)
        elif(model_name == 'vnet'):
            self.model = vnet_2d(**model_params)
        elif(model_name == 'unet_plus'):
            self.model = unet_plus_2d(**model_params)
        elif(model_name == 'r2_unet'):
            self.model = r2_unet_2d(**model_params)
        elif(model_name == 'attunet'):
            self.model = att_unet_2d(**model_params)
        elif(model_name == 'resunet'):
            self.model = resunet_a_2d(**model_params)
        elif(model_name == 'u2_net'):
            self.model = u2net_2d(**model_params)
        elif(model_name == 'unet3_plus'):
            self.model = unet_3plus_2d(**model_params)
        elif(model_name == 'transunet'):
            self.model = transunet_2d(**model_params)
        elif(model_name == 'swinunet'):
            self.model = swin_unet_2d(**model_params)
        else:
            self.model = None

        if(os.path.isdir('results')):
            shutil.rmtree('results')
        os.mkdir('results')

    def create_callbacks(self):
        # name_seed = np.random.randint(0,16,12)
        # model_hash = ''.join([str(hex(i))[2] for i in name_seed])

        self.adam = tensorflow.keras.optimizers.Adam(learning_rate=self.train_params['learning_rate'])
        
        self.callbacks = []
        self.callbacks += [tensorflow.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',mode='max',
            min_delta=0.001,patience=self.train_params['callback_params']['patience'],
            restore_best_weights=True)]
        
        # lr_exp < 1: expoential decay
        # lr_exp >=1: stepwise decay at every 
        def scheduler(epoch,lr):
            if self.train_params['callback_params']['lr_exp'] < 1:
                if epoch < self.train_params['callback_params']['lr_decay_start']:
                    return lr
                else:
                    return lr * self.train_params['callback_params']['lr_exp']
            else:
                if (epoch+1) % self.train_params['callback_params']['lr_decay_start'] == 0:
                    return lr / self.train_params['callback_params']['lr_exp']
                else:
                    return lr

        self.callbacks += [tensorflow.keras.callbacks.LearningRateScheduler(scheduler)]
        self.callbacks += [cb]
        # self.callbacks += [tensorflow.keras.callbacks.ModelCheckpoint(
        #                     model_hash,
        #                     monitor=['val_loss','val_accuracy'],
        #                     verbose=1,
        #                     save_best_only=False,
        #                     save_weights_only=False,
        #                     mode="auto",
        #                     save_freq="epoch",
        #                     options=None)]

    def train(self,tr_data,tr_label,tr_coords,ts_data,ts_label,ts_coords,num_classes):
        self.create_callbacks()

        tr_label_cat = to_categorical(tr_label,num_classes)
        ts_label_cat = to_categorical(ts_label,num_classes)

        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.lr
            return lr
        lr_metric = get_lr_metric(self.adam)

        if self.train_params['loss'] == 'cce':
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif self.train_params['loss'] == 'crps2d':
            loss = crps2d_tf
        elif self.train_params['loss'] == 'dice-c':
            loss = dice_coef
        elif self.train_params['loss'] == 'dice':
            loss = dice
        elif self.train_params['loss'] == 'tversky':
            loss = tversky
        elif self.train_params['loss'] == 'focal-t':
            loss = focal_tversky
        elif self.train_params['loss'] == 'iou-seg':
            loss = iou_seg

        self.model.compile(optimizer=self.adam,loss=loss,metrics = ['categorical_accuracy',lr_metric])
        self.model_history = self.model.fit(tr_data,tr_label_cat,epochs=self.train_params['epochs'],
            batch_size=self.train_params['batch_size'],use_multiprocessing=True,workers=8,
            validation_data=(ts_data,ts_label_cat),verbose=1,
            callbacks=self.callbacks)

    def evaluate(self,ts_data,ts_label,ts_coords):
        plt.semilogy(self.model_history.history['loss'])
        plt.semilogy(self.model_history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} - Loss')
        plt.legend(['Train','Validation'])
        plt.savefig(f'results/loss_{self.model_name}.png',dpi=300,bbox_inches='tight')
        plt.show()

        pred_ts = self.model.predict(ts_data,batch_size = self.train_params['batch_size'])
        pred_ts_prob = np.stack([reconstruct_image(pred_ts[:,:,:,i],ts_coords,256) for i in range(self.model_params['num_classes'])],axis=2)
        plt.imsave(f'results/prob1_ts_{self.model_name}.png',pred_ts_prob[:,:,:3],vmin=0,vmax=1)
        plt.imsave(f'results/prob2_ts_{self.model_name}.png',pred_ts_prob[:,:,[3,3,4]],vmin=0,vmax=1)

        self.pred_cl = np.argmax(pred_ts,axis=3).astype(np.int8)
        self.acc = np.mean(ts_label == self.pred_cl)
        print(f'{self.model_name} accuracy: {self.acc}')

        cmap_article = ListedColormap([
            [0.008, 0, 0, 1],
            [0.921, 0.822, 0.729, 1], #Agriculture
            [0.2, 0.592, 0.2, 1],     #Forest
            [0.896, 0.312, 0, 1],     #Built-up
            [0.1, 0.6, 0.93, 1]])  #Water

        # cmap_article = ListedColormap([
        #     [0.008, 0, 0, 1],
        #     [0.871, 0.722, 0.529, 1],
        #     [0, 0.392, 0, 1],
        #     [0.796, 0.012, 0, 1],
        #     [0.004, 0, 0.392, 1]])

        cmap_article_s = LinearSegmentedColormap.from_list("",[
            [0.008, 0, 0, 1],
            [0.871, 0.722, 0.529, 1],
            [0, 0.392, 0, 1],
            [0.796, 0.012, 0, 1],
            [0.004, 0, 0.392, 1]])

        cmap_me = ListedColormap([
            [0.008, 0, 0, 1],
            [0.871, 0.722, 0.529, 1],
            [0.13, 0.55, 0.13, 1],
            [0.4, 0.4, 0.4, 1],
            [0.09, 0.45, 0.80, 1]])

        pred_img = reconstruct_image(self.pred_cl,ts_coords,self.model_params['input_size'][0])

        plt.figure(figsize=(16,8))
        plt.imshow(pred_img[::5,::5],cmap=cmap_article)
        plt.imsave(f'results/label_ts_{self.model_name}_cmap_art.png',pred_img,vmin=0,vmax=self.model_params['num_classes'],cmap=cmap_article)
        plt.imsave(f'results/label_ts_{self.model_name}_cmap_me.png',pred_img,vmin=0,vmax=self.model_params['num_classes'],cmap=cmap_me)
        plt.show()

        # dill.dump(self,gzip.open(f'results/model_{self.model_name}.pkl.gz','wb'))
        dill.dump(self.model_history.history,open('results/model_history.pkl','wb'))

        self.metrics = calculate_metrics(ts_label.flatten()[::10],self.pred_cl.flatten()[::10])
        print(self.metrics[:-1])
        print(self.metrics[-1])
        dill.dump(self.metrics,open('results/metrics.pkl','wb'))

        with open('results/model_summary.txt', 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        dill.dump(cb.logs,open('results/train_times.pkl','wb'))
        dill.dump(self.train_params,open('results/train_params.pkl','wb'))
        dill.dump(self.model_params,open('results/model_params.pkl','wb'))


    def save_model(self):
        now = datetime.now().strftime('%Y%m%d-%H%M%S')
        folder_name = f'{self.dataset_name}-{self.model_name}-{self.train_params["loss"]}-{now}-{int(self.acc*1e4)}'
        print(f'Moving files to {folder_name}')
        shutil.move('results',folder_name)
        return folder_name

    def save_model_colab(self):
        now = datetime.now().strftime('%Y%m%d-%H%M')

        folder_name = f'{now}_{self.dataset_name}_{self.train_params["learning_rate"]:.0e}_{self.train_params["callback_params"]["lr_exp"]}_{self.train_params["loss"]}_{self.model_name}_{int(self.acc*1e4)}'
        print(f'Moving files to {folder_name}')
        shutil.move('results','/content/drive/MyDrive/runs/'+folder_name)
        return folder_name