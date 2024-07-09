'''
This module implements the loss function for WisdomLearning to optimize feature integration and extraction.

Author: [Saeed Iqbal]
Date: [December 26, 2023]

Usage:
    # Example usage of WisdomLearningLoss
    loss_function = WisdomLearningLoss()
    loss = loss_function.compute_loss(predictions, labels)
    print(loss)

Notes:
    This implementation defines the loss function used during training for WisdomLearning.
'''
class WisdomLearningLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(WisdomLearningLoss, self).__init__()

    def call(self, fwisdom, y_true):
        return tf.reduce_mean(tf.square(fwisdom - y_true))

