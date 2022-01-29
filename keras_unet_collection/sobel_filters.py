from keras.layers import Dense, Embedding, Conv1D, Conv2D, Lambda, Input, concatenate
from keras.layers import Dropout, Conv3D, Activation, Concatenate, DepthwiseConv2D
from keras.optimizers import adam_v2
from keras.models import Model # 这里我们学习使用Model型的模型
from tensorflow.keras.utils import to_categorical
# .utils import to_categorical
import keras.backend as K # 引入Keras后端来自定义loss，注意Keras模型内的一切运算
                          # 必须要通过Keras后端完成，比如取对数要用K.log不能用np.log
# from tensorflow.python.keras.layers import Lambda, Convolution2D;


def sobel_x_initializer(shape, dtype=None):
    print(shape)    
    sobel_x = tf.constant(
        [
            [-1, -2, -1], 
            [0, 0, 0], 
            [1, 2, 1]
        ], dtype=dtype )
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))

    print(tf.shape(sobel_x))
    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2],shape[-1]))

    print(tf.shape(sobel_x))
    return sobel_x

def sobel_y_initializer(shape, dtype=None):
    print(shape)    
    sobel_y = tf.constant(
        [
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]
        ], dtype=dtype )
    #create the missing dims.
    sobel_y = tf.reshape(sobel_y, (3, 3, 1, 1))

    print(tf.shape(sobel_y))
    #tile the last 2 axis to get the expected dims.
    sobel_y = tf.tile(sobel_y, (1, 1, shape[-2],shape[-1]))

    print(tf.shape(sobel_y))
    return sobel_y