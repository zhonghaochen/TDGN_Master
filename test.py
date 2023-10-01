import tensorflow as tf
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import numpy as np
from GRU import ConvGRU2D
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

# test_data_1 = np.load("./dataset1/XtestfarmTestRatio0.8.npy")
# test_data_2 = np.load("./dataset2/XtestfarmTestRatio0.8.npy")
# test_label = np.load("./dataset2/ytestfarmTestRatio0.8.npy")
# all_data_1 = np.load("./dataset1/allDatafarmTestRatio0.8.npy")
# all_data_2 = np.load("./dataset2/allDatafarmTestRatio0.8.npy")
# all_label = np.load("./dataset2/allLabelsfarmTestRatio0.8.npy")
# labels = sio.loadmat('../Datasets/farm450/label.mat')['label']
# model = tf.keras.models.load_model('./model/CD_model_farm.hdf5', custom_objects={'ConvGRU2D': ConvGRU2D})

# test_data_1 = np.load("./dataset1/XtestriverTestRatio0.8.npy")
# test_data_2 = np.load("./dataset2/XtestriverTestRatio0.8.npy")
# test_label = np.load("./dataset2/ytestriverTestRatio0.8.npy")
# all_data_1 = np.load("./dataset1/allDatariverTestRatio0.8.npy")
# all_data_2 = np.load("./dataset2/allDatariverTestRatio0.8.npy")
# all_label = np.load("./dataset2/allLabelsriverTestRatio0.8.npy")
# labels = sio.loadmat('../Datasets/river/label.mat')['label']
# model = tf.keras.models.load_model('./model/CD_model_river.hdf5', custom_objects={'ConvGRU2D': ConvGRU2D})

# test_data_1 = np.load("./dataset1/XtestHermistonTestRatio0.8.npy")
# test_data_2 = np.load("./dataset2/XtestHermistonTestRatio0.8.npy")
# test_label = np.load("./dataset2/ytestHermistonTestRatio0.8.npy")
# all_data_1 = np.load("./dataset1/allDataHermistonTestRatio0.8.npy")
# all_data_2 = np.load("./dataset2/allDataHermistonTestRatio0.8.npy")
# all_label = np.load("./dataset2/allLabelsHermistonTestRatio0.8.npy")
# labels = sio.loadmat('../Datasets/Hermiston/label.mat')['label']
# model = tf.keras.models.load_model('./model/CD_model_Hermiston.hdf5', custom_objects={'ConvGRU2D': ConvGRU2D})

test_data_1 = np.load("./dataset1/XtestUSATestRatio0.8.npy")
test_data_2 = np.load("./dataset2/XtestUSATestRatio0.8.npy")
test_label = np.load("./dataset2/ytestUSATestRatio0.8.npy")
all_data_1 = np.load("./dataset1/allDataUSATestRatio0.8.npy")
all_data_2 = np.load("./dataset2/allDataUSATestRatio0.8.npy")
all_label = np.load("./dataset2/allLabelsUSATestRatio0.8.npy")
labels = sio.loadmat('../Datasets/USA/label.mat')['label']
model = tf.keras.models.load_model('./model/CD_model_USA3.hdf5', custom_objects={'ConvGRU2D': ConvGRU2D})

predicted = model.predict([test_data_1, test_data_2])
predicted = np.argmax(predicted, 1)
test_label = test_label[:, 1]

TN, FP, FN, TP = confusion_matrix(test_label.flatten(), predicted.flatten()).ravel()
OA = (TP + TN) / (TP + TN + FP + FN)
P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2 * P * R / (R + P)
PRE = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / ((TP + TN + FP + FN) ** 2)
KC = (OA - PRE) / (1 - PRE)
print('Accuracy: {}\nRecall: {}\nPr: {}\nF1-Score: {}\nKappa: {}'.format(OA, R, P, F1, KC))
print('Predict Finished')
with open('result/testmetrics.txt', 'w') as file:
    file.write('{} Test Accuracy'.format(OA*100))
    file.write('\n')
    file.write('{} Kappa'.format(KC))
    file.write('\n')
    file.write('{} F1-Score'.format(F1*100))
    file.write('\n')
    file.write('{} Pr'.format(P*100))
    file.write('\n')
    file.write('{} Recall'.format(R*100))
    file.write('\n')
    file.write('{} TP'.format(TP))
    file.write('\n')
    file.write('{} TN'.format(TN))
    file.write('\n')
    file.write('{} FP'.format(FP))
    file.write('\n')
    file.write('{} FN'.format(FN))

all_predicted = model.predict([all_data_1, all_data_2])
all_predicted = np.argmax(all_predicted, 1)
all_label = all_label[:, 1]
all_predicted = np.reshape(all_predicted, [labels.shape[0], labels.shape[1]])
plt.subplot(1, 1, 1)
plt.imshow(all_predicted, cmap='Greys_r')
plt.xticks([])
plt.yticks([])
plt.savefig('predict.png', dpi=900, pad_inches=0.0)
# io.imsave('预测图.png', np.uint8(all_predicted * 255))
print('Valar Dohaeris')