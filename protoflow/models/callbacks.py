import numpy as np
import tensorflow as tf
import datetime as dt
import gc
import os

from ..models.vis import plot_space, gen_gifs


def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.005)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)


class GramSchmidt(tf.keras.callbacks.Callback):
    def __init__(self,
                 omega_layer=None):
        super(GramSchmidt, self).__init__()
        self.omega_layer = omega_layer

    def gram_schmidt(self, A):
        """Orthogonalize a set of vectors stored as the columns of matrix A."""
        # Get the number of vectors.
        n = A.shape[1]
        for j in range(n):
            # To orthogonalize the vector in column j with respect to the
            # previous vectors, subtract from it its projection onto
            # each of the previous vectors.
            for k in range(j):
                A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
            if np.linalg.norm(A[:, j]) != 0.0:
                A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
            else:
                A[:, j] = A[:, j] / (np.linalg.norm(A[:, j]) + 1e-10)
        return A

    def omega_normalization(self, Ω):
        nors = tf.sqrt(
                tf.linalg.trace(tf.matmul(tf.transpose(Ω), Ω)))
        return Ω / nors
        
    # def on_train_batch_end(self, _, logs=None):   
    def on_train_epoch_end(self, _, logs=None):
        new_weights = self.gram_schmidt(self.omega_layer.get_weights()[0])
        new_weights = self.omega_normalization(new_weights)
        new_weights = np.expand_dims(new_weights, axis=0)
        self.omega_layer.set_weights(new_weights)


class AlphaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, max_epochs, mode='sigmoid'):
        super(AlphaScheduler, self).__init__()
        self.mode = mode
        self.max_epochs = max_epochs

    def on_epoch_begin(self, e, logs=None):
        if self.mode == 'exp':
            self.model.layers[-1].alpha.assign(-tf.math.exp(e/3))
        elif self.mode == 'sigmoid':
            """
            fg_e, slowness = 50, 10
            i = (e - fg_e) / slowness
            #self.alpha.assign(1 - 1/(1+tf.math.exp(-i)))
            self.model.layers[-1].alpha.assign(1/(1+tf.math.exp(-i)))
            """
            shift = 1/3 # 1 times high alpha, 2 times low alpha
            slope = 4
            upper_offset = 0.0
            lower_offset = 0.05
            scaling = 1 - (lower_offset + upper_offset)
            sigmoidal = (1 - upper_offset) - scaling * (1 / (1 + tf.math.exp(-(e/self.max_epochs - shift) * slope)))
            self.model.layers[-1].alpha.assign(sigmoidal)
        else:
            pass
        return


class PlotTraining(tf.keras.callbacks.Callback):
    def __init__(self, x, y_classes, y_sources, tgmlvq):
        self.x = x
        self.yc = y_classes
        self.ys = y_sources
        self.tgmlvq = tgmlvq
        self.e = 0
        self.t = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        self.img_path = f"imgs/{self.t}/"
        if not os.path.isdir("imgs/"):
            os.mkdir("imgs/")
        os.mkdir(self.img_path)
        os.mkdir(self.img_path+"classes/")
        os.mkdir(self.img_path+"sources/")
    
    def on_epoch_begin(self, e, logs=None):
        self.e = e

    def on_train_batch_end(self, b, logs=None):
        img_name = f"{self.e:04d}{b:04d}.png"
        #print(logs['loss'], logs['src_acc'], logs['cls_acc'])
        plot_space(self.x, self.yc, self.tgmlvq, "null", img_name, self.img_path+"sources/", "source-GMLVQ", acc=logs['cls_acc'])
        plot_space(self.x, self.ys, self.tgmlvq, "row", img_name, self.img_path+"classes/", "class-GMLVQ", acc=logs['src_acc'])
        gc.collect()

    def on_train_end(self, logs=None):
        gen_gifs(self.t)
