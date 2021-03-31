#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 21:00:22 2021

@author: gam
"""
import tensorflow as tf

def ce_custom_loss(y_true, y_pred):
    val = tf.square(y_true) * tf.math.log(y_pred) + y_true - y_pred
    return tf.reduce_sum(-val, -1)


# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
def compare_models(history1, history2):
    # list all data in history
    print(history1.history.keys())
    # summarize history for accuracy
    plt.plot(history1.history['acc'])
    plt.plot(history1.history['val_acc'])
    plt.plot(history2.history['acc'])
    plt.plot(history2.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['ce train', 'ce test', 'ce+ train', 'ce+ test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['ce train', 'ce test', 'ce+ train', 'ce+ test'], loc='upper left')
    plt.show()