# General collecting part of all the models

from . import model_segnet
from . import model_danet
from .keras_unet.models.custom_unet import custom_unet
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.utils import to_categorical

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
            self.model = model_danet.danet_resnet101(model_params['image_shape'][0],model_params['image_shape'][1],
                model_params['image_shape'][2],model_params['num_classes'])
        else:
            self.model = None

    def create_callbacks(self):
        self.callbacks = []
        self.callbacks += [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=self.train_params['callback_params']['patience'],restore_best_weights=True)]
        
        def scheduler(epoch,lr):
            if epoch < self.train_params['callback_params']['lr_decay_start']:
                return lr
            else:
                return lr * self.train_params['callback_params']['lr_exp']

        self.callbacks += [tensorflow.keras.callbacks.LearningRateScheduler(scheduler)]

    def train(self,tr_data,tr_label,val_data,val_label):
        self.create_callbacks()

        tr_label_cat = to_categorical(tr_label,self.model_params['num_classes'])
        val_label_cat = to_categorical(val_label,self.model_params['num_classes'])

        self.model.compile(optimizer='Adam',loss=tensorflow.losses.CategoricalCrossentropy())
        self.model_history = self.model.fit(tr_data,tr_label_cat,epochs=self.train_params['epochs'],
            batch_size=self.train_params['batch_size'],use_multiprocessing=True,workers=8,
            validation_data=(val_data,val_label_cat),verbose=1,
            callbacks=self.callbacks)

        plt.semilogy(self.model_history.history['loss'])
        plt.semilogy(self.model_history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} - Loss')
        plt.legend(['Train','Validation'])
        plt.show()

    def evaluate(self):
        model.acc = 0.5
        pass

    def save(self):
        pass