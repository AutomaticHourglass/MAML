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
        elif(model_name == 'vnet'):
            self.model = vnet_2d(**model_params)
        elif(model_name == 'unet_plus'):
            self.model = unet_plus_2d(**model_params)
        elif(model_name == 'r2_unet'):
            self.model = r2_unet_2d(**model_params)
        elif(model_name == 'att_unet'):
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
        self.adam = tensorflow.keras.optimizers.Adam(learning_rate=self.train_params['learning_rate'])
        
        self.callbacks = []
        self.callbacks += [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=self.train_params['callback_params']['patience'],restore_best_weights=True)]
        
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
        elif self.train_params['loss'] == 'crps2d_tf':
            loss = crps2d_tf
        elif self.train_params['loss'] == 'crps2d_np':
            loss = crps2d_np
        elif self.train_params['loss'] == 'dice_coef':
            loss = dice_coef
        elif self.train_params['loss'] == 'dice':
            loss = dice
        elif self.train_params['loss'] == 'tversky_coef':
            loss = tversky_coef
        elif self.train_params['loss'] == 'tversky':
            loss = tversky
        elif self.train_params['loss'] == 'focal_tversky':
            loss = focal_tversky
        elif self.train_params['loss'] == 'ms_ssim':
            loss = ms_ssim
        elif self.train_params['loss'] == 'iou_box_coef':
            loss = iou_box_coef
        elif self.train_params['loss'] == 'iou_box':
            loss = iou_box
        elif self.train_params['loss'] == 'iou_seg':
            loss = iou_seg
        elif self.train_params['loss'] == 'triplet_1d':
            loss = triplet_1d


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

        pred_ts = self.model.predict(ts_data)
        self.pred_cl = np.argmax(pred_ts,axis=3).astype(np.int8)
        self.acc = np.mean(ts_label == self.pred_cl)
        print(f'{self.model_name} accuracy: {self.acc}')

        pred_img = reconstruct_image(self.pred_cl,ts_coords,self.model_params['input_size'][0])
        plt.imshow(pred_img[::5,::5],)
        plt.imsave(f'results/label_ts_{self.model_name}.png',pred_img,vmin=0,vmax=self.model_params['num_classes'])
        plt.show()

        dill.dump(self,gzip.open(f'results/model_{self.model_name}.pkl.gz','wb'))

    def save_model(self):
        now = datetime.now().strftime('%Y%m%d-%H%M%S')
        folder_name = f'{self.dataset_name}-{self.model_name}-{self.train_params["loss"]}-{now}-{int(self.acc*1e4)}'
        print(f'Moving files to {folder_name}')
        shutil.move('results',folder_name)
        return folder_name