import numpy as np 
import os

import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as kbe
from tensorflow.keras import losses
from tensorflow.keras import metrics

from tensorflow.math import multiply, reduce_mean

def dice_coef(y_true, y_pred, smooth=1):
    """
    :param smooth Prevent zero division error
    :param y_true: Train ground truth
    :param y_pred: Predicted
    :return: Returns the dice coefficient
    """

    # greater = kbe.greater_equal(y_pred, 0.5) #will return boolean values
    # y_pred_binary = kbe.cast(greater, dtype=kbe.floatx()) #will convert bool to 0 and 1  

    y_true_f = kbe.flatten(y_true)
    y_pred_f = kbe.flatten(y_pred)
    intersection = kbe.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (kbe.sum(y_true_f) + kbe.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    :param y_true: Train ground truth
    :param y_pred: Predicted
    :return: Returns dice loss
    """
    return 1 - dice_coef(y_true, y_pred)


def pixel_weighted_cross_entropy(weights, targets, predictions):
    pixel_losses = losses.binary_crossentropy(targets, predictions)
    #print(pixel_losses)
    squeezed_weights = reduce_mean(weights, axis=3, name='drop_one_dim_weights')
    #print(squeezed_weights)
    weighted_loss_val = multiply(squeezed_weights,
                                 pixel_losses,
                                 name='multiply_pixelwise_bce_by_weights'
                                )
    #print(weighted_loss_val)
    output_loss = reduce_mean(weighted_loss_val, name='average_weighted_pixel_losses')

    #combined_loss = 0.5 * (output_loss + dice_coef_loss(targets, predictions))

    return output_loss
    #return combined_loss


# def loss_wrapper(weights):
#     def pixel_weighted_cross_entropy(targets, predictions):
#         loss_val = losses.binary_crossentropy(targets, predictions)
#         weighted_loss_val = weights * loss_val
#         return kbe.mean(weighted_loss_val)
#     return pixel_weighted_cross_entropy


def unet(pretrained_weights = None, input_shape = (256,256,1)):
    inputs = Input(input_shape)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(learning_rate=1e-4, use_ema=False, ema_momentum=.99), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-4), loss=[dice_coef_loss], metrics=[dice_coef])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def weighted_loss_unet(pretrained_weights=None, input_shape=(256, 256, 1), inference=False):
    
    inputs = Input(shape=input_shape, name="inputs")
    
    if not inference:
        targets = Input(shape=input_shape, name="targets")
        print(targets)
        weights_ip = Input(shape=input_shape, name="weights")
        print(weights_ip)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid', name = "outputs")(conv9)

    ## old (i.e. without pixel weighting):
    # model = Model(inputs = inputs, outputs = conv10)
    # model.compile(optimizer = Adam(learning_rate=1e-4, use_ema=False, ema_momentum=.99), loss = 'binary_crossentropy', metrics = ['accuracy'])

    if not inference:
        model = Model(inputs=(inputs, targets, weights_ip), outputs=conv10)
        model.add_loss(pixel_weighted_cross_entropy(weights_ip, targets, conv10))

    else:
        model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(learning_rate=1e-4, use_ema=False, ema_momentum=.99), metrics=[metrics.BinaryAccuracy(), metrics.MeanIoU(num_classes=2)], run_eagerly=False)
    #model.compile(optimizer=Adam(learning_rate=1e-4), metrics=[metrics.BinaryAccuracy(), metrics.MeanIoU(num_classes=2)])
    

    ## does not work:  
    # model = Model(inputs=(inputs, weights_ip), outputs=conv10)
    # model.compile(loss=loss_wrapper(weights_ip), optimizer=Adam(learning_rate=1e-4, use_ema=False, ema_momentum=.99), metrics=['accuracy'])
    
    #model.compile(optimizer = Adam(lr = 1e-4), loss=[dice_coef_loss], metrics=[dice_coef])

    if not inference:
        model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model





def unet_minimal(pretrained_weights = None, input_shape = (256,256,1)):
    inputs = Input(input_shape)

    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)

    up4 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop3))
    merge4 = concatenate([conv2, up4], axis = 3)
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv4 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    up5 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge5 = concatenate([conv1, up5], axis = 3)
    conv5 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    conv6 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    conv7 = Conv2D(1, 1, activation = 'sigmoid')(conv6)

    model = Model(inputs = inputs, outputs = conv7)

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-4), loss=[dice_coef_loss], metrics=[dice_coef])

    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
