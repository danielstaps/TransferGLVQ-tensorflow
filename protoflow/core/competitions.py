import itertools
import numpy as np
import tensorflow as tf


def squared_euclidean_distance(x, y):
    expanded_x = tf.expand_dims(x, axis=1)
    batchwise_difference = tf.subtract(y, expanded_x)
    differences_raised = tf.math.pow(batchwise_difference, 2)
    distances = tf.reduce_sum(input_tensor=differences_raised, axis=2)
    return distances


def accuracy_score(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    match = y_true == y_pred
    accuracy = tf.reduce_mean(tf.cast(match, 'float32'))
    return accuracy


def accuracy_score_missing_labels(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    match = y_true == y_pred
    minus_one = tf.constant(float(-1))
    num_no_label = tf.reduce_sum(tf.cast(tf.equal(y_true, minus_one), 'float32'))
    num_classifi = tf.cast(tf.shape(y_true)[0], 'float32')
    sum_classifi = tf.reduce_sum(tf.cast(match, 'float32'))
    accuracy = sum_classifi / (num_classifi - num_no_label) 
    return accuracy


def wtac_accuracy(prototype_labels):
    """Returns a Winner-Takes-All-Competition-based accuracy metric function"""
    def acc(y_true, distances):
        winning_indices = tf.keras.backend.argmin(distances, axis=1)
        y_pred = tf.gather(prototype_labels, winning_indices)
        accuracy = accuracy_score(tf.reshape(y_true, shape=(-1, )), y_pred)
        return accuracy
    return acc


def wtac_accuracy_twotasks(name, prototype_labels, pps, nsources, source=True):
    """Returns a Winner-Takes-All-Competition-based accuracy metric function"""
    def acc(y_true, distances):
        sp = pps * nsources
        source_y_true, class_y_true = y_true[:,0], y_true[:,1]
        source_distances, class_distances = distances[:,:sp], distances[:,sp:]
        if source:
            y_true = source_y_true
            distances = source_distances
        else:
            y_true = class_y_true
            distances = class_distances
        y_true = tf.expand_dims(y_true, axis=-1)
        winning_indices = tf.keras.backend.argmin(distances, axis=1)
        y_pred = tf.gather(prototype_labels, winning_indices)
        accuracy = accuracy_score_missing_labels(
                tf.reshape(y_true, shape=(-1, )), y_pred)
        return accuracy
    acc.__name__ = name
    return acc
