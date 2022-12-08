import tensorflow as T
from tensorflow.keras import backend as K
from tensorflow.keras.losses import *


# custom loss functions for missing values in input data (i.e. target labels or values)

MISSING_LABEL_FLAG = -1
BinaryCrossentropy = T.keras.losses.BinaryCrossentropy()


def classification_loss(loss_function, mask_value=MISSING_LABEL_FLAG): #used in multitask classification models (customizable)
    
    """Builds a loss function that masks based on targets
    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets
    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):  
        
        y_true = T.where(T.math.is_nan(y_true), T.zeros_like(y_true), y_true)
        dtype = K.floatx()
        mask = T.cast(T.not_equal(y_true, MISSING_LABEL_FLAG), dtype)
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function


def regression_loss(y_true, y_pred):  #Used in multitask regression models (MSE metric)
    y_true = T.where(T.math.is_nan(y_true), T.zeros_like(y_true), y_true)
    loss = T.reduce_mean(
       T.where(T.equal(y_true, 0.0), y_true,
       T.square(T.abs(y_true - y_pred))))
       
    return loss
    
        
# adaptative learning rate using during training

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
        
    return lr