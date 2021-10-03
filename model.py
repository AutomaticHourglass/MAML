# General collecting part of all the models

# import model_segnet
# import model_danet
from .keras_unet.models.custom_unet import custom_unet

class SSegModel:
    def __init__(self,model_name,model_params):
        if(model_name == 'unet'):
            self.model = custom_unet(**model_params)