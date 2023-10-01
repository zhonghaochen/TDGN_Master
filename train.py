import os
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_depth_5 import *
import datetime
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

model = model()
model.summary()

lr = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.001, decay_rate=0.1, decay_steps=150)
Adam = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='loss', patience=80, verbose=1, mode='auto')

# train_data_1 = np.load("./dataset1/XtrainfarmTestRatio0.8.npy")
# train_data_2 = np.load("./dataset2/XtrainfarmTestRatio0.8.npy")
# train_label = np.load("./dataset2/ytrainfarmTestRatio0.8.npy")
# checkpoint = ModelCheckpoint('./model/CD_model_farm.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')

# train_data_1 = np.load("./dataset1/XtrainriverTestRatio0.8.npy")
# train_data_2 = np.load("./dataset2/XtrainriverTestRatio0.8.npy")
# train_label = np.load("./dataset2/ytrainriverTestRatio0.8.npy")
# checkpoint = ModelCheckpoint('./model/CD_model_river.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')

# train_data_1 = np.load("./dataset1/XtrainHermistonTestRatio0.8.npy")
# train_data_2 = np.load("./dataset2/XtrainHermistonTestRatio0.8.npy")
# train_label = np.load("./dataset2/ytrainHermistonTestRatio0.8.npy")
# checkpoint = ModelCheckpoint('./model/CD_model_Hermiston.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')

train_data_1 = np.load("./dataset1/XtrainUSATestRatio0.8.npy")
train_data_2 = np.load("./dataset2/XtrainUSATestRatio0.8.npy")
train_label = np.load("./dataset2/ytrainUSATestRatio0.8.npy")
checkpoint = ModelCheckpoint('./model/CD_model_USA.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [earlyStopping, checkpoint]
start_time = datetime.datetime.now()
history = model.fit([train_data_1, train_data_2], train_label, batch_size=64, epochs=200, shuffle=True, verbose=1, callbacks=callbacks_list)
end_time = datetime.datetime.now()
train_time = end_time - start_time
print('START -----{}-----\nEND -----{}-----\nCOST -----{}-----'.format(start_time, end_time, train_time))
print('Valar Morghulis')






