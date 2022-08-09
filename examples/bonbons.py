import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# DATA
from protoflow.datasets.get_data import get_data
from protoflow.datasets.prepare_data import (perturbate_data,
                                   prepare_data_for_crossvalidation)

# T-GLVQ
from protoflow.models.glvq import GLVQ, GMLVQ, TGLVQ
from protoflow.models.callbacks import (AlphaScheduler, GramSchmidt, PlotTraining,
                              lr_callback)

from protoflow.core.losses import GLVQLoss, TGLVQLoss
from protoflow.core.competitions import wtac_accuracy, wtac_accuracy_twotasks


# plotting
from protoflow.models.vis import plot_3d, plot_pca, plot_space, print_paras


DATASET = "bonbons"

SOURCES_PROTOTYPES_PER_CLASS=1
CLASSES_PROTOTYPES_PER_CLASS=1
MAPPING_DIMENSION=10
K = 5



""" data """
ox, oy_classes, oy_sources, nclasses, nsources = get_data(dataset=DATASET)
print("\n=== shapes ===")
print(f"data      : {ox.shape}")
print(f"y_classes : {oy_classes.shape} [{nclasses} classes]")
print(f"y_sources : {oy_sources.shape}\n")


plot_3d(ox, oy_classes, "classes")
plot_3d(ox, oy_sources, "sources")
#plot_pca(x, y_classes, "pca of data/classes")

x, y_classes, y_sources = prepare_data_for_crossvalidation(ox, oy_classes, nclasses, k=K, y_sources=oy_sources)
#for i in range(len(x)):
for i in range(1):
    ((x_train, x_test), (yc_train, yc_test), (ys_train, ys_test)) = perturbate_data(
            x, y_classes, i, K, y_sources)
    print(ys_train, yc_train)
    print(ys_test, yc_test)
    print(nclasses, nsources)

    """ model """
    inputs = tf.keras.Input(shape=(x_train.shape[1], ), name='inputs')
    tglvq = TGLVQ(vpc=MAPPING_DIMENSION,
                  pps=SOURCES_PROTOTYPES_PER_CLASS,
                  ppc=CLASSES_PROTOTYPES_PER_CLASS,
                  nclasses=nclasses,
                  nsources=nsources,
                  name='tglvq')
    tglvq_model = tf.keras.Model(inputs=inputs,
            outputs=tglvq(inputs),
            name='T-GLVQ_model')


    tglvq_model.summary()
    tglvq.omega_layer.init(x_train)

    #tglvq.source_prototype_layer.init_mean(x_train, ys_train)
    #tglvq.source_prototype_layer.init_random(x_train, ys_train)
    tglvq.source_prototype_layer.init_random_fromoneclass(x_train, yc_train, ys_train)      
    
    #tglvq.class_prototype_layer.init_mean(x_train, yc_train)
    #tglvq.class_prototype_layer.init_random(x_train, yc_train)
    tglvq.class_prototype_layer.init_random_fromonesource(x_train, yc_train, ys_train)

    #print_paras(tglvq)

    source_wtac_metric = wtac_accuracy_twotasks(
            'src_acc',
            tglvq.source_prototype_layer.prototype_labels,
            pps=SOURCES_PROTOTYPES_PER_CLASS,
            nsources=nsources,
            source=True)
    class_wtac_metric = wtac_accuracy_twotasks(
            'cls_acc',
            tglvq.class_prototype_layer.prototype_labels,
            pps=SOURCES_PROTOTYPES_PER_CLASS,
            nsources=nsources,
            source=False)

    loss_fn = TGLVQLoss(tglvq.source_prototype_layer.prototype_labels,
                        tglvq.class_prototype_layer.prototype_labels,
                        pps=SOURCES_PROTOTYPES_PER_CLASS,
                        nsources=nsources,
                        alpha_source=tglvq.alpha) # 1

    learning_rate = 0.1
    tglvq_model.compile(loss=loss_fn,
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=[source_wtac_metric, class_wtac_metric],
            run_eagerly=True)

    EPOCHS = 200
    tglvq_model.fit(
            x_train,
            np.stack([yc_train, ys_train], axis=-1),
            #batch_size=(x_train.shape[0]//20)-1,
            batch_size=x_train.shape[0],
            epochs=EPOCHS,
            callbacks=[GramSchmidt(tglvq.omega_layer),
                       #PlotTraining(ox, oy_sources, oy_classes, tglvq), # TODO
                       lr_callback,
                       AlphaScheduler(max_epochs=EPOCHS, mode='sigmoid')
                      ])

    print("\nEvaluate full data set:")
    evals = tglvq_model.evaluate(x_test, np.stack([yc_test, ys_test], axis=-1))
    print(evals)
    print("\nEvaluation of every source:")
    for ns in range(nsources):
        indices = np.where(ys_test == ns)
        x_test_set = x_test[indices]
        ys_test_set = ys_test[indices]
        yc_test_set = yc_test[indices]
        evals = tglvq_model.evaluate(x_test_set, np.stack([yc_test_set, ys_test_set], axis=-1))
        print(evals)

    #print_paras(tglvq)


    #plot_pca(x, y_classes, "pca of Ω mapped data/classes")
    #plot_space_3d(ox, oy_classes, tglvq, "null")
    #plot_space_3d(ox, oy_sources, tglvq, "row")
    #plot_class_space_2d(x_train, yc_train, tglvq)
