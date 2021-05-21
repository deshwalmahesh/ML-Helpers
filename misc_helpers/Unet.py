import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import random
import pickle


SEED = 13
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

batch_size = 8
epochs = 1
w, h = 1280, 224
dir = '../fine_tune/images/'
optimizer = 'adam'


def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


reg = regularizers.l2(0.001)

metrics = [r_squared,tf.keras.metrics.MeanSquaredError(name='mse_metric'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse_metric'),
            tf.keras.metrics.MeanAbsoluteError(name='mae_metric')]


def get_model(pretrained_weights = None,input_size = (h,w,1)):
    inputs = Input(input_size)
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

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def run_all_models():
    # ---------- MSE LOSS ----------
    gen = ImageDataGenerator(rescale=1./255.,rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,
                            horizontal_flip=True,vertical_flip=True,zoom_range=0.1,shear_range=0.1,
                            validation_split=0.13,)

    train = gen.flow_from_directory(dir,color_mode='grayscale',class_mode='input',batch_size=batch_size,target_size=(h,w),seed=SEED,subset='training')
    val = gen.flow_from_directory(dir,color_mode='grayscale',class_mode='input',batch_size=batch_size,target_size=(h,w),seed=SEED,subset='validation')

    es = EarlyStopping(verbose=1,restore_best_weights=True,patience=5)
    mcp = ModelCheckpoint('MSE.h5',save_best_only=True,verbose=1,mode='min',save_weights_only=True)
    rlp = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=3, min_lr=0.0001)


    keras.backend.clear_session()
    model = get_model()
    model.compile(optimizer=optimizer,loss='mse',metrics=metrics)
    history = model.fit(train,epochs=epochs,steps_per_epoch=len(train)//batch_size,validation_data=val,
                        callbacks=[es,mcp,rlp])

    with open('MSE_hist.pkl', 'wb') as f:
        pickle.dump(history.history, f)


    # ---------- RMSE LOSS ----------

    gen = ImageDataGenerator(rescale=1./255.,rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,
                            horizontal_flip=True,vertical_flip=True,zoom_range=0.1,shear_range=0.1,
                            validation_split=0.13,)

    train = gen.flow_from_directory(dir,color_mode='grayscale',class_mode='input',batch_size=batch_size,target_size=(h,w),seed=SEED,subset='training')
    val = gen.flow_from_directory(dir,color_mode='grayscale',class_mode='input',batch_size=batch_size,target_size=(h,w),seed=SEED,subset='validation')

    es = EarlyStopping(verbose=1,restore_best_weights=True,patience=5)
    mcp = ModelCheckpoint('RMSE.h5',save_best_only=True,verbose=1,mode='min',save_weights_only=True)
    rlp = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=3, min_lr=0.0001)

    keras.backend.clear_session()

    model = get_model()
    model.compile(optimizer=optimizer,loss=rmse,metrics=metrics)

    history = model.fit(train,epochs=epochs,steps_per_epoch=len(train)//batch_size,validation_data=val,
                        callbacks=[es,mcp,rlp])

    with open('RMSE_hist.pkl', 'wb') as f:
        pickle.dump(history.history, f)


    # ---------- MAE LOSS ----------

    gen = ImageDataGenerator(rescale=1./255.,rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,
                            horizontal_flip=True,vertical_flip=True,zoom_range=0.1,shear_range=0.1,
                            validation_split=0.13,)

    train = gen.flow_from_directory(dir,color_mode='grayscale',class_mode='input',batch_size=batch_size,target_size=(h,w),seed=SEED,subset='training')
    val = gen.flow_from_directory(dir,color_mode='grayscale',class_mode='input',batch_size=batch_size,target_size=(h,w),seed=SEED,subset='validation')

    es = EarlyStopping(verbose=1,restore_best_weights=True,patience=5)
    mcp = ModelCheckpoint('MAE.h5',save_best_only=True,verbose=1,mode='min',save_weights_only=True)
    rlp = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=3, min_lr=0.0001)

    keras.backend.clear_session()

    model = get_model()
    model.compile(optimizer=optimizer,loss='mae',metrics=metrics)

    history = model.fit(train,epochs=epochs,steps_per_epoch=len(train)//batch_size,validation_data=val,
                        callbacks=[es,mcp,rlp])

    with open('MAE_hist.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # ----------- BINARY CrossEntropy -------------------

    gen = ImageDataGenerator(rescale=1./255.,rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,
                            horizontal_flip=True,vertical_flip=True,zoom_range=0.1,shear_range=0.1,
                            validation_split=0.13,)

    train = gen.flow_from_directory(dir,color_mode='grayscale',class_mode='input',batch_size=batch_size,target_size=(h,w),seed=SEED,subset='training')
    val = gen.flow_from_directory(dir,color_mode='grayscale',class_mode='input',batch_size=batch_size,target_size=(h,w),seed=SEED,subset='validation')

    es = EarlyStopping(verbose=1,restore_best_weights=True,patience=5)
    mcp = ModelCheckpoint('Bin_Cross.h5',save_best_only=True,verbose=1,mode='min',save_weights_only=True)
    rlp = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=3, min_lr=0.0001)

    keras.backend.clear_session()

    model = get_model()
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=metrics)

    history = model.fit(train,epochs=epochs,steps_per_epoch=len(train)//batch_size,validation_data=val,
                        callbacks=[es,mcp,rlp])

    with open('Bin_Cross_hist.pkl', 'wb') as f:
        pickle.dump(history.history, f)

# run_all_models()