# General collecting part of all the models

from . import model_segnet
from . import model_danet
from .keras_unet.models.custom_unet import custom_unet

class SSegModel:
    def __init__(self,model_name,model_params):
        self.model_name = model_name
        self.model_params = model_params

        if(model_name == 'unet'):
            self.model = custom_unet(**model_params)
        elif(model_name) == 'segnet':
            self.model = model_segnet.segnet(**model_params)
        elif(model_name) == 'danet':
            self.model = model_danet.danet_resnet101(**model_params)
        else:
            self.model = None