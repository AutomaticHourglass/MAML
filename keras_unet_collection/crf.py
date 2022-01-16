# -*- coding:utf-8 -*-

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

import tensorflow as tf
import tensorflow_addons as tfa

class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        outputs = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs * labels, 2), 1, keepdims=True)  # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans * labels, [2, 3]), 1, keepdims=True)
        return point_score + trans_score  # 两部分得分之和

    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred):  # 目标y_pred需要是one hot形式
        if self.ignore_last_label:
            mask = 1 - y_true[:, :, -1:]
        else:
            mask = K.ones_like(y_pred[:, :, :1])
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
        init_states = [y_pred[:, 0]]  # 初始状态
        y_pred = K.concatenate([y_pred, mask])
        log_norm, _, _ = K.rnn(self.log_norm_step, y_pred[:, 1:], init_states)  # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        return log_norm - path_score  # 即log(分子/分母)

    def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1 - y_true[:, :, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)


"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def _unpack_data(data):
    """Support sample_weight"""
    if len(data) == 3:
        x, y, sample_weight = data
    else:
        x, y, sample_weight = data[0], data[1], None
    return x, y, sample_weight


class CRFModel(tf.keras.Model):

    def __init__(self,
                 model: tf.keras.Model,
                 units: int,
                 chain_initializer="orthogonal",
                 use_boundary: bool = True,
                 boundary_initializer="zeros",
                 use_kernel: bool = True,
                 **kwargs):
        # build functional model
        crf = tfa.layers.CRF(
            units=units,
            chain_initializer=chain_initializer,
            use_boundary=use_boundary,
            boundary_initializer=boundary_initializer,
            use_kernel=use_kernel,
            **kwargs)
        # take model's first output passed to CRF layer
        decode_sequence, potentials, sequence_length, kernel = crf(inputs=model.outputs[0])
        # set name for outputs
        decode_sequence = tf.keras.layers.Lambda(lambda x: x, name='decode_sequence')(decode_sequence)
        potentials = tf.keras.layers.Lambda(lambda x: x, name='potentials')(potentials)
        sequence_length = tf.keras.layers.Lambda(lambda x: x, name='sequence_length')(sequence_length)
        kernel = tf.keras.layers.Lambda(lambda x: x, name='kernel')(kernel)
        super().__init__(
            inputs=model.inputs,
            outputs=[decode_sequence, potentials, sequence_length, kernel],
            **kwargs)
        self.crf = crf

    def train_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        if isinstance(y, dict):
            y = list(y.values())[0]
        with tf.GradientTape() as tape:
            decode_sequence, potentials, sequence_length, kernel = self(x, training=True)
            crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
            if sample_weight is not None:
                crf_loss = crf_loss * sample_weight
            crf_loss = tf.reduce_mean(crf_loss)
            loss = crf_loss + sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results

    def test_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        if isinstance(y, dict):
            y = list(y.values())[0]
        decode_sequence, potentials, sequence_length, kernel = self(x, training=False)
        crf_loss = -tfa.text.crf_log_likelihood(potentials, y, sequence_length, kernel)[0]
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        crf_loss = tf.reduce_mean(crf_loss)
        loss = crf_loss + sum(self.losses)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, potentials)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results


class CRFNew(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, mask, **kwargs):
        self.mask = mask
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        super(CRFNew, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1]
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        # inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        outputs = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = self.mask * outputs + (1 - self.mask) * states[:, :, 0]
        return outputs, [outputs]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs * labels, 2), 1, keepdims=True)  # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans * labels, [2, 3]), 1, keepdims=True)
        return point_score + trans_score  # 两部分得分之和

    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred):  # 目标y_pred需要是one hot形式
        # y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
        init_states = y_pred  # 初始状态
        y_pred = K.concatenate([y_pred, self.mask])
        log_norm, _, _ = K.rnn(self.log_norm_step, y_pred, self.mask)  # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        return log_norm - path_score  # 即log(分子/分母)

    def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        # y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if self.mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * self.mask) / K.sum(self.mask)
