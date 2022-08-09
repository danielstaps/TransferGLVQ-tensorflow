import itertools
import numpy as np
import tensorflow as tf


class Prototypes1D(tf.keras.layers.Layer):
    def __init__(self,
                 prototypes_per_class=1,
                 nclasses=None,
                 prototype_initializer='zeros',
                 trainable_prototypes=True,
                 dtype='float32',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_data'), )
        super().__init__(dtype=dtype, **kwargs)

        self.nclasses = nclasses
        self.num_of_prototypes = prototypes_per_class * nclasses
        self.prototype_distribution = [prototypes_per_class] * nclasses
        self.prototype_initializer = prototype_initializer
        self.trainable_prototypes = trainable_prototypes

    def build(self, input_shape):
        # Make a label list and flatten the list of lists using itertools
        llist = [[i] * n
                 for i, n in zip(range(len(self.prototype_distribution)),
                                 self.prototype_distribution)]
        flat_llist = list(itertools.chain(*llist))
        self.prototype_labels = tf.Variable(initial_value=flat_llist,
                                            dtype=self.dtype,
                                            trainable=False)
        self.prototypes = self.add_weight(
            name='prototypes',
            shape=(self.num_of_prototypes, input_shape[1]),
            dtype=self.dtype,
            initializer=self.prototype_initializer,
            trainable=self.trainable_prototypes)
        super().build(input_shape)

    def call(self, x):
        return self.prototypes

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_of_prototypes)

    def shift_protos(self, prototypes):
        random_normal = np.random.normal(loc=0., scale=0.1, size=prototypes.shape)
        return np.add(prototypes, random_normal)


    def init_mean(self, x_train, y_train):
        idc = [np.where(y_train == class_) for class_ in range(self.nclasses)]
        new_proto_w = []
        for class_ in self.prototype_labels.numpy():
            class_mean = np.mean(x_train[idc[int(class_)]], axis=0)
            new_proto_w.append(class_mean)
        new_proto_w = self.shift_protos(np.asarray(new_proto_w))
        act_w = self.get_weights()
        act_w[0] = new_proto_w
        self.set_weights(act_w)

    def init_random(self, x_train, y_train):
        idc = [np.where(y_train == class_) for class_ in range(self.nclasses)]
        new_proto_w = []
        for class_ in self.prototype_labels.numpy():
            xc = x_train[idc[int(class_)]]
            i = np.random.randint(low=0, high=len(xc))
            new_proto_w.append(xc[i])
        new_proto_w = self.shift_protos(np.asarray(new_proto_w))
        act_w = self.get_weights()
        act_w[0] = new_proto_w
        self.set_weights(act_w)

    def init_random_fromonesource(self, x_train, yc, ys):
        idc = [np.where(ys == source_) for source_ in range(len(set(ys)))]
        xs = x_train[idc[0]]
        yc = yc[idc[0]]
        self.init_random(xs, yc)

    def init_random_fromoneclass(self, x_train, yc, ys):
        self.init_random_fromonesource(x_train, ys, yc)



class OmegaMatrix(tf.keras.layers.Layer):
    def __init__(self, units_mapped_dim, **kwargs):
        super(OmegaMatrix, self).__init__(**kwargs)
        self.units_mapped_dim = units_mapped_dim

    def build(self, input_shape):
        self.units_input_dim = input_shape[-1]
        self.omega_matrix = self.add_weight(
                name='omega_matrix',
                shape=(self.units_mapped_dim, self.units_input_dim),
                initializer='random_normal',
                trainable=True)

    def call(self, x):
        return tf.matmul(x, tf.transpose(self.omega_matrix))

    def get_eigValVec(self, data):
        data_cov_mat = np.cov(data.T)
        eigVal, eigVec = np.linalg.eig(data_cov_mat)
        eigVal = eigVal/np.sum(eigVal)
        eigVec = np.asarray(eigVec, dtype='float32')
        idx = np.abs(eigVal).argsort()[::-1] # sorts descending
        eigVal, eigVec = eigVal[idx], eigVec[idx]
        return eigVal, eigVec

    def init(self, x):
        if self.units_mapped_dim == self.units_input_dim:
            identity = np.identity(x.shape[1])
            new_weights = np.expand_dims(identity, axis=0)
        else:
            eigVal, eigVec = self.get_eigValVec(x)
            eigVec = eigVec[:,:self.units_mapped_dim]
            eigVec = eigVec.T
            new_weights = np.expand_dims(eigVec, axis=0)
        self.set_weights(new_weights)

    def get_inverse(self, ):
        if self.units_mapped_dim == self.units_input_dim:
            return tf.linalg.inv(self.omega_matrix)
        else:
            # get_moore_penrose_inverse
            return tf.linalg.pinv(self.omega_matrix)

    def classification_correlation_matrix(self, ):
        return tf.matmul(tf.transpose(self.omega_matrix), self.omega_matrix)

    def regularization(self, X, P):
        eigVal, eigVec = self.get_eigValVec(np.concatenate((X, P), axis=0))
        MCA = eigVec[np.argmin(eigVal<0.01):]
        Phi = np.eye(P.shape[1]) - np.dot(MCA, MCA)
        new_omega_weights = np.matmul(Phi.T, np.squeeze(self.get_weights()))
        print(new_omega_weights.shape)
        self.set_weights(np.expand_dims(new_omega_weights, axis=0))


