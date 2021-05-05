import tensorflow as tf

from abc import abstractmethod


class Metric:
    def __init__(self, num_classes, is_binary_cross_entropy=False):
        """
        Metrics are counted (Recall, Precision, F1Score).

        :param num_classes: number of classes in the dataset.
        :param is_binary_cross_entropy: If there are no more than two classes, the value is set to True.
        """
        self.num_classes = num_classes
        self.epsilon = 1e-6
        self.is_binary_cross_entropy = is_binary_cross_entropy
        self.__name__ = 'metric'
        if self.is_binary_cross_entropy and self.num_classes != 2:
            msg = 'There should be 2 classes with binary cross entropy, got {}.'.format(self.num_classes)
            raise ValueError(msg)

    def confusion_matrix(self, y_true, y_pred):
        """
        This functions counts confusion_matrix.

        :param y_true: This is the true mark of validation data.
        :param y_pred: This is the predict mark of validation data.
        :return: False Positive, False Negative, True Positive.
        """
        if self.is_binary_cross_entropy:
            y_true = tf.cast(y_true > 0.5, tf.float32)[:, 0]
            y_pred = tf.cast(y_pred > 0.5, tf.float32)[:, 0]
        else:
            y_true = tf.argmax(y_true, 1)
            y_pred = tf.argmax(y_pred, 1)

        matrix = tf.cast(tf.math.confusion_matrix(y_true, y_pred, self.num_classes), tf.float32)
        fp = tf.reduce_sum(matrix, axis=0) - tf.linalg.tensor_diag_part(matrix)
        fn = tf.reduce_sum(matrix, axis=1) - tf.linalg.tensor_diag_part(matrix)
        tp = tf.linalg.tensor_diag_part(matrix)
        return fp, fn, tp

    @abstractmethod
    def __call__(self, y_true, y_pred):
        raise NotImplementedError('This method must be implemented in subclasses')


class Recall(Metric):
    def __init__(self, num_classes, is_binary_cross_entropy=False):
        super().__init__(num_classes, is_binary_cross_entropy)
        self.__name__ = 'recall'

    def __call__(self, y_true, y_pred):
        fp, fn, tp = self.confusion_matrix(y_true, y_pred)
        return tp / (tp + fn + self.epsilon)


class Precision(Metric):
    def __init__(self, num_classes, is_binary_cross_entropy=False):
        super().__init__(num_classes, is_binary_cross_entropy)
        self.__name__ = 'precision'

    def __call__(self, y_true, y_pred):
        fp, fn, tp = self.confusion_matrix(y_true, y_pred)
        return tp / (tp + fp + self.epsilon)


class F1Score(Metric):
    def __init__(self, num_classes, is_binary_cross_entropy=False, beta=1):
        super().__init__(num_classes, is_binary_cross_entropy)
        self.beta = beta
        self.__name__ = 'F1_score'

    def __call__(self, y_true, y_pred):
        fp, fn, tp = self.confusion_matrix(y_true, y_pred)
        recall = tp / (tp + fn + self.epsilon)
        precision = tp / (tp + fp + self.epsilon)
        return (self.beta ** 2 + 1) * precision * recall / (self.beta ** 2 * precision + recall + self.epsilon)
