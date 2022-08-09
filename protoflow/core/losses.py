import tensorflow as tf


class GLVQLoss(tf.keras.losses.Loss):
    def __init__(self, prototype_labels, **kwargs):
        self.prototype_labels = prototype_labels
        super().__init__(**kwargs)

    def call(self, y_true, distances):
        y_true = tf.cast(y_true, self.prototype_labels.dtype)
        matching = tf.equal(y_true, self.prototype_labels)
        unmatching = tf.logical_not(matching)

        inf = tf.constant(float('inf'))
        d_matching = tf.where(matching, distances, inf)
        d_unmatching = tf.where(unmatching, distances, inf)
        dp = tf.keras.backend.min(d_matching, axis=1, keepdims=True)
        dm = tf.keras.backend.min(d_unmatching, axis=1, keepdims=True)

        mu = (dp - dm) / (dp + dm)
        loss = tf.keras.backend.sum(mu, axis=0)
        return loss


class TGLVQLoss(tf.keras.losses.Loss):
    def __init__(self, source_prototype_labels,
                 class_prototype_labels, pps, nsources, alpha_source, **kwargs):
        self.source_prototype_labels = source_prototype_labels
        self.class_prototype_labels = class_prototype_labels
        self.alpha_source = alpha_source
        self.sp = pps * nsources
        super().__init__(**kwargs)

    def call(self, y_true, distances):
        source_y_true, class_y_true = y_true[:,0], y_true[:,1]
        source_y_true = tf.expand_dims(source_y_true, axis=-1)
        class_y_true = tf.expand_dims(class_y_true, axis=-1)
        source_distances, class_distances = distances[:,:self.sp], distances[:,self.sp:]
        source_y_true = tf.cast(source_y_true, self.source_prototype_labels.dtype)
        matching = tf.equal(source_y_true, self.source_prototype_labels)
        unmatching = tf.logical_not(matching)

        inf = tf.constant(float('inf'))
        max_d = tf.keras.backend.max(source_distances, axis=[0,1], keepdims=False)
        d_matching = tf.where(matching, source_distances, max_d)
        d_unmatching = tf.where(unmatching, source_distances, max_d)
        dp = tf.keras.backend.min(d_matching, axis=1, keepdims=True)
        dm = tf.keras.backend.min(d_unmatching, axis=1, keepdims=True)

        mu = (dp - dm) / (dp + dm)
        source_loss = tf.keras.backend.sum(mu, axis=0)


        class_y_true = tf.cast(class_y_true, self.class_prototype_labels.dtype)
        matching = tf.equal(class_y_true, self.class_prototype_labels)
        unmatching = tf.logical_not(matching)
        
        
        inf = tf.constant(float('inf'))
        max_d = tf.keras.backend.max(class_distances, axis=[0,1], keepdims=False)
        d_matching = tf.where(matching, class_distances, max_d)
        d_unmatching = tf.where(unmatching, class_distances, max_d)
        dp = tf.keras.backend.min(d_matching, axis=1, keepdims=True)
        dm = tf.keras.backend.min(d_unmatching, axis=1, keepdims=True)

        nu = (dp - dm) / (dp + dm)
        class_loss = tf.keras.backend.sum(nu, axis=0)

        loss = self.alpha_source * source_loss + (1-self.alpha_source) * class_loss
        return loss
