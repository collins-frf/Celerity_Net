# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

def pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error)
    loss.__name__ = "pixel_error"

def absolute_error(y_true, y_pred):
    error = y_pred - y_true
    abs_error = tf.math.abs(error)
    return tf.reduce_mean(abs_error)
    loss.__name__ = "absolute_error"

def pred_max(target, output):
    return tf.reduce_max(output)
    loss.__name__ = "pred_max"

def pred_min(target, output):
    return tf.reduce_min(output)
    loss.__name__ = "pred_min"

def huber(y_true, y_pred):
    delta = 1
    #_epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    #y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    #y_true = tf.cast(y_true, tf.float32)

    loss = np.where(np.abs(y_true-y_pred) < delta,
        0.5*((y_true-y_pred)**2),
        delta*np.abs(y_true - y_pred) - 0.5*(delta**2))
    
    return np.sum(loss)
    loss.__name__ = "huber"
