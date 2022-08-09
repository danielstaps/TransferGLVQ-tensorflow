import tensorflow as tf

from ..core.components import Prototypes1D, OmegaMatrix
from ..core.competitions import squared_euclidean_distance


class GLVQ(tf.keras.models.Model):
    def __init__(self, ppc=1, nclasses=10, **kwargs):
        super().__init__(**kwargs)
        self.prototype_layer = Prototypes1D(
                prototypes_per_class=ppc,
                nclasses=nclasses,
                trainable_prototypes=True,
                prototype_initializer="random_uniform",
                name="prototype_layer")

    def call(self, inputs):
        prototypes = self.prototype_layer(inputs)
        distances = squared_euclidean_distance(inputs, prototypes)
        # Optional regularization to encourage the prototypes to be close to data
        # reg1 = 1 * tf.reduce_mean(tf.keras.backend.min(distances, axis=1))
        # self.add_loss(reg1)
        return distances


class GMLVQ(tf.keras.models.Model):
    def __init__(self, vpc=1, ppc=1, nclasses=10, **kwargs):
        super().__init__(**kwargs)
        self.prototype_layer = Prototypes1D(
                prototypes_per_class=ppc,
                nclasses=nclasses,
                trainable_prototypes=True,
                prototype_initializer="random_uniform",
                name="prototype_layer")
        self.omega_matrix = OmegaMatrix(vpc)

    def call(self, inputs):
        prototypes = self.prototype_layer(inputs)
        input_omega_map = self.omega_matrix(inputs)
        proto_omega_map = self.omega_matrix(prototypes)
        distances = squared_euclidean_distance(input_omega_map, proto_omega_map)
        # Optional regularization to encourage the prototypes to be close to data
        # reg1 = 1 * tf.reduce_mean(tf.keras.backend.min(distances, axis=1))
        # self.add_loss(reg1)
        return distances


class TGLVQ(tf.keras.models.Model):
    def __init__(self, vpc=1, pps=1, ppc=1, nclasses=10, nsources=2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.Variable(1.)
        self.source_prototype_layer = Prototypes1D(
                prototypes_per_class=pps,
                nclasses=nsources,
                trainable_prototypes=True,
                prototype_initializer="random_uniform",
                name="source_prototype_layer")
        self.class_prototype_layer = Prototypes1D(
                prototypes_per_class=ppc,
                nclasses=nclasses,
                trainable_prototypes=True,
                prototype_initializer="random_uniform",
                name="class_prototype_layer")
        self.omega_layer = OmegaMatrix(vpc)
        """ vis """
        self.omega_layer_2 = OmegaMatrix(1)
    
    def ComplementaryOrthogonalProjector(self, ):
        return tf.eye(self.omega_layer.omega_matrix.shape[1]) - tf.matmul(
                tf.transpose(self.omega_layer.omega_matrix), 
                self.omega_layer.omega_matrix)
        
    def call(self, inputs):
        """source"""
        source_prototypes = self.source_prototype_layer(inputs)
        input_omega_map = self.omega_layer(inputs)
        proto_omega_map = self.omega_layer(source_prototypes)
        source_distances = squared_euclidean_distance(input_omega_map, proto_omega_map)
        
        """class"""
        class_prototypes = self.class_prototype_layer(inputs)
        input_COP_map = tf.matmul(inputs, tf.transpose(
            self.ComplementaryOrthogonalProjector()))
        proto_COP_map = tf.matmul(class_prototypes, tf.transpose(
            self.ComplementaryOrthogonalProjector()))
        """ vis """
        input_COP_map = self.omega_layer_2(input_COP_map)
        proto_COP_map = self.omega_layer_2(proto_COP_map)
        """ vis end """
        class_distances = squared_euclidean_distance(input_COP_map, proto_COP_map)


        # Optional regularization to encourage the prototypes to be close to data
        # reg1 = 1 * tf.reduce_mean(tf.keras.backend.min(distances, axis=1))
        # self.add_loss(reg1)
        stack = tf.concat([source_distances, class_distances], axis=1)
        return stack
