# General collecting part of all the models

from . import model_segnet
from . import model_danet
from .keras_unet.models.custom_unet import custom_unet
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.utils import to_categorical
import numpy as np
from .common import *
import pickle
import gzip

class SSegModel:
    def __init__(self,model_name,model_params,train_params):
        self.model_name = model_name
        self.model_params = model_params
        self.train_params = train_params

        if(model_name == 'unet'):
            self.model = custom_unet(**model_params)
        elif(model_name) == 'segnet':
            self.model = model_segnet.segnet(**model_params)
        elif(model_name) == 'danet':
            self.model = model_danet.danet_resnet101(**model_params)
        else:
            self.model = None

    def create_callbacks(self):
        self.adam = tensorflow.keras.optimizers.Adam(learning_rate=self.train_params['learning_rate'])
        
        self.callbacks = []
        self.callbacks += [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=self.train_params['callback_params']['patience'],restore_best_weights=True)]
        
        def scheduler(epoch,lr):
            if epoch < self.train_params['callback_params']['lr_decay_start']:
                return lr
            else:
                return lr * self.train_params['callback_params']['lr_exp']

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

        self.model.compile(optimizer=self.adam,loss=tensorflow.losses.CategoricalCrossentropy(),metrics = ['categorical_accuracy',lr_metric])
        self.model_history = self.model.fit(tr_data,tr_label_cat,epochs=self.train_params['epochs'],
            batch_size=self.train_params['batch_size'],use_multiprocessing=True,workers=8,
            validation_data=(ts_data,ts_label_cat),verbose=1,
            callbacks=self.callbacks)

        plt.semilogy(self.model_history.history['loss'])
        plt.semilogy(self.model_history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} - Loss')
        plt.legend(['Train','Validation'])
        plt.savefig(f'loss_{self.model_name}.png',dpi=300,bbox_inches='tight')
        plt.show()

    def evaluate(self,ts_data,ts_label,ts_coords):
        self.pred_ts = self.model.predict(ts_data)
        self.pred_cl = np.argmax(self.pred_ts,axis=3)

        self.acc = np.mean(ts_label == self.pred_cl)
        print(f'{self.model_name} accuracy: {self.acc}')

        self.pred_img = reconstruct_image(self.pred_cl,ts_coords,self.model_params['input_shape'][0])
        plt.imshow(self.pred_img[::5,::5],)
        plt.imsave(f'label_ts_{self.model_name}.png',self.pred_img,vmin=0,vmax=self.model_params['num_classes'])
        plt.show()

        pickle.dump(self,gzip.open(f'model_{self.model_name}.pkl.gz','wb'))
        
    def save(self):
        pass