import os
import random
import numpy as np
import scipy
import scipy.io as sio
import scipy.ndimage
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def data_load(Datasets_name):
    if Datasets_name == 'farm':
        data_1 = sio.loadmat('../Datasets/farm450/farm06.mat')['imgh']
        data_2 = sio.loadmat('../Datasets/farm450/farm07.mat')['imghl']
        labels = sio.loadmat('../Datasets/farm450/label.mat')['label']

    if Datasets_name == 'river':
        data_1 = sio.loadmat('../Datasets/river/river_before.mat')['river_before']
        data_2 = sio.loadmat('../Datasets/river/river_after.mat')['river_after']
        labels = sio.loadmat('../Datasets/river/label.mat')['label']

    if Datasets_name == 'Hermiston':
        data_1 = sio.loadmat('../Datasets/Hermiston/hermiston2004.mat')['HypeRvieW']
        data_2 = sio.loadmat('../Datasets/Hermiston/hermiston2007.mat')['HypeRvieW']
        labels = sio.loadmat('../Datasets/Hermiston/label.mat')['label']

    if Datasets_name == 'USA':
        data_1 = sio.loadmat('../Datasets/USA/USA_before.mat')['USA_before']
        data_2 = sio.loadmat('../Datasets/USA/USA_after.mat')['USA_after']
        labels = sio.loadmat('../Datasets/USA/label.mat')['label']

    return data_1, data_2, labels

def apply_pca(x, num_components):
    new_x = np.reshape(x, (-1,x.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_x = pca.fit_transform(new_x)
    new_x = np.reshape(new_x, (x.shape[0], x.shape[1], num_components))
    return new_x

def pad_with_zeros(x, margin=3):
    new_x = np.zeros((x.shape[0] + 2 * margin, x.shape[1] + 2 * margin, x.shape[2]))
    new_x[margin:x.shape[0] + margin, margin:x.shape[1] + margin, :] = x
    return new_x

def create_patches(x, y, patch_size):
    margin = int((patch_size - 1) / 2)
    zero_patches_x = pad_with_zeros(x, margin)
    patches_data = np.zeros((x.shape[0] * x.shape[1], patch_size, patch_size, x.shape[2]))
    patches_label = np.zeros((x.shape[0] * x.shape[1]))
    patch_index = 0
    for c in range(x.shape[0]):
        for r in range(x.shape[1]):
            patches_data[patch_index, :, :, :] = zero_patches_x[c:c + patch_size, r:r + patch_size, :]
            patches_label[patch_index] = y[c, r]
            patch_index += 1
    return patches_data, patches_label

def augment_data(x, y):
    x_aug = np.zeros_like(x)
    y_aug = y
    for i in range(x.shape[0]):
        patch = x[i, :, :, :]
        num = random.randint(0, 3)
        flipped_patch = patch
        if num == 0:
            flipped_patch = np.flipud(patch)
        if num == 1:
            flipped_patch = np.fliplr(patch)
        if num == 2:
            no = random.randrange(-180, 180, 30)
            flipped_patch = scipy.ndimage.rotate(patch, no, axes=(1, 0),
                                                 reshape=False, output=None, order=3, mode='constant',
                                                 cval=0.0, prefilter=False)
        x_aug[i, :, :, :] = flipped_patch
    new_x = np.concatenate((x, x_aug))
    new_y = np.concatenate((y, y_aug))
    return new_x, new_y

def set_train_sample(x, y, pos, neg):
    np.random.seed(138)
    rand_perm = np.random.permutation(y.shape[0])
    new_x = x[rand_perm, :, :, :]
    new_y = y[rand_perm]

    train_x0 = new_x[new_y == 0, :, :, :][:neg]
    train_y0 = new_y[new_y == 0][:neg]
    train_x1 = new_x[new_y == 1, :, :, :][:pos]
    train_y1 = new_y[new_y == 1][:pos]

    test_x0 = new_x[new_y == 0, :, :, :][neg:]
    test_y0 = new_y[new_y == 0][neg:]
    test_x1 = new_x[new_y == 1, :, :, :][pos:]
    test_y1 = new_y[new_y == 1][pos:]

    x_train = np.concatenate((train_x0, train_x1))
    y_train = np.concatenate((train_y0, train_y1))
    x_test = np.concatenate((test_x0, test_x1))
    y_test = np.concatenate((test_y0, test_y1))
    return  x_train, x_test, y_train, y_test

def split_train_test_set(x, y, test_ratio):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=321, stratify=y)
    return x_train, x_test, y_train, y_test


def SaveX_trainPatches_1(X_trainPatches, TestRatio, Datasets_name=''):
    with open("./dataset1/Xtrain" + Datasets_name + "TestRatio" + str(TestRatio) + ".npy", "bw") as outfile:
        np.save(outfile, X_trainPatches)
def SaveX_testPatches_1(X_testPatches, TestRatio, Datasets_name=''):
    with open("./dataset1/Xtest" + Datasets_name + "TestRatio" + str(TestRatio) + ".npy", "bw") as outfile:
        np.save(outfile, X_testPatches)
def Saveall_data_1(all_data, TestRatio, Datasets_name=''):
    with open("./dataset1/allData" + Datasets_name + "TestRatio" + str(TestRatio) + ".npy", "bw") as outfile:
        np.save(outfile, all_data)

def SaveX_trainPatches_2(X_trainPatches, TestRatio, Datasets_name=''):
    with open("./dataset2/Xtrain" + Datasets_name + "TestRatio" + str(TestRatio) + ".npy", "bw") as outfile:
        np.save(outfile, X_trainPatches)
def SaveX_testPatches_2(X_testPatches, TestRatio, Datasets_name=''):
    with open("./dataset2/Xtest" + Datasets_name + "TestRatio" + str(TestRatio) + ".npy", "bw") as outfile:
        np.save(outfile, X_testPatches)
def Savey_trainPatches(y_trainPatches, TestRatio, Datasets_name=''):
    with open("./dataset2/ytrain" + Datasets_name + "TestRatio" + str(TestRatio) + ".npy", "bw") as outfile:
        np.save(outfile, y_trainPatches)
def Savey_testPatches(y_testPatches, TestRatio, Datasets_name=''):
    with open("./dataset2/ytest" + Datasets_name + "TestRatio" + str(TestRatio) + ".npy", "bw") as outfile:
        np.save(outfile, y_testPatches)
def Saveall_data_2(all_data, TestRatio, Datasets_name=''):
    with open("./dataset2/allData" + Datasets_name + "TestRatio" + str(TestRatio) + ".npy", "bw") as outfile:
        np.save(outfile, all_data)
def Saveall_labels(all_labels, TestRatio, Datasets_name=''):
    with open("./dataset2/allLabels" + Datasets_name + "TestRatio" + str(TestRatio) + ".npy", "bw") as outfile:
        np.save(outfile, all_labels)




patch_size = 7
Test_ratio = 0.8

# Datasets_name = 'farm'
# pos = 4400
# neg = 8800

# Datasets_name = 'river'
# pos = 1250
# neg = 2500

# Datasets_name = 'Hermiston'
# pos = 2600
# neg = 5200

Datasets_name = 'USA'
pos = 3313
neg = 3919


x_1, x_2, y = data_load(Datasets_name=Datasets_name)
bands = x_1.shape[2]


if __name__ =='__main__':
    x_patch_1, y_1 = create_patches(x_1, y, patch_size=patch_size)
    # x_train_1, x_test_1, y_train_1, y_test_1 = split_train_test_set(x_patch_1, y_1, test_ratio=Test_ratio)
    x_train_1, x_test_1, y_train_1, y_test_1 = set_train_sample(x_patch_1, y_1, pos=pos, neg=neg)
    # train_data_1, y_train_1 = augment_data(x_train_1, y_train_1)
    train_data_1, y_train_1 = x_train_1, y_train_1
    train_data_1 = (train_data_1 / np.float32(np.max(train_data_1)))
    SaveX_trainPatches_1(train_data_1, TestRatio=Test_ratio, Datasets_name=Datasets_name)
    test_data_1 = (x_test_1 / np.float32(np.max(x_test_1)))
    SaveX_testPatches_1(test_data_1, TestRatio=Test_ratio, Datasets_name=Datasets_name)
    data_1 = (x_patch_1 / np.float32(np.max(x_patch_1)))
    Saveall_data_1(data_1, TestRatio=Test_ratio, Datasets_name=Datasets_name)

    x_patch_2, y_2 = create_patches(x_2, y, patch_size=patch_size)
    # x_train_2, x_test_2, y_train_2, y_test_2 = split_train_test_set(x_patch_2, y_2, test_ratio=Test_ratio)
    x_train_2, x_test_2, y_train_2, y_test_2 = set_train_sample(x_patch_2, y_2, pos=pos, neg=neg)
    # train_data_2, y_train_2 = augment_data(x_train_2, y_train_2)
    train_data_2, y_train_2 = x_train_2, y_train_2
    train_label = np_utils.to_categorical(y_train_2)
    Savey_trainPatches(train_label, TestRatio=Test_ratio, Datasets_name=Datasets_name)
    train_data_2 = (train_data_2 / np.float32(np.max(train_data_2)))
    SaveX_trainPatches_2(train_data_2, TestRatio=Test_ratio, Datasets_name=Datasets_name)
    test_label = np_utils.to_categorical(y_test_2)
    Savey_testPatches(test_label, TestRatio=Test_ratio, Datasets_name=Datasets_name)
    test_data_2 = (x_test_2 / np.float32(np.max(x_test_2)))
    SaveX_testPatches_2(test_data_2, TestRatio=Test_ratio, Datasets_name=Datasets_name)
    label = np_utils.to_categorical(y_2)
    Saveall_labels(label, TestRatio=Test_ratio, Datasets_name=Datasets_name)
    data_2 = (x_patch_2 / np.float32(np.max(x_patch_2)))
    Saveall_data_2(data_2, TestRatio=Test_ratio, Datasets_name=Datasets_name)









