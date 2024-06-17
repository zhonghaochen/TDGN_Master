import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from GRU import ConvGRU2D
from Datapre_new_new import *

patch_size = args.patches
bands = data_t1.shape[2]


def Spectral_extraction(x, i):
    x = kl.Conv3D(filters=i * 8, kernel_size=(1, 1, 5), strides=(1, 1, 2))(x)
    x = kl.BatchNormalization()(x)
    x = kl.ReLU()(x)
    return x

def Spatial_extraction(x, i):
    band = x.shape[3]
    x = kl.Conv3D(filters=i * 8, kernel_size=(1, 1, band))(x)
    x = kl.BatchNormalization()(x)
    x = kl.ReLU()(x)
    x = kl.Permute((3, 1, 2, 4))(x)
    x = kl.Conv2D(filters=i * 8, kernel_size=(3, 3), padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.ReLU()(x)
    return x

def ConvGRU(x_1, x_2, i):
    x_GRU = kl.concatenate((x_1, x_2), axis=1)
    x_GRU = ConvGRU2D(filters=i * 8, kernel_size=(3, 3), padding='same', return_sequences=False)(x_GRU)
    return x_GRU


def Differential_guided_attention(x_GRU, x_spe):
    x_GRU = kl.Activation('relu')(x_GRU)
    spatial_feature = kl.Reshape((x_GRU.shape[1], x_GRU.shape[2], 1, x_GRU.shape[3]))(x_GRU)
    x = kl.multiply([spatial_feature, x_spe])
    # x = kl.add([x, x_spe])
    return x


def model():
    Ta_0 = kl.Input(shape=(patch_size, patch_size, bands))
    Tb_0 = kl.Input(shape=(patch_size, patch_size, bands))

    x_1 = kl.Reshape((Ta_0.shape[1], Ta_0.shape[2], Ta_0.shape[3], 1))(Ta_0)
    x_2 = kl.Reshape((Tb_0.shape[1], Tb_0.shape[2], Tb_0.shape[3], 1))(Tb_0)

    '''光谱提取'''
    Ta_1_spe = Spectral_extraction(x_1, 1)
    Tb_1_spe = Spectral_extraction(x_2, 1)

    Ta_1_spa = Spatial_extraction(Ta_1_spe, 1)
    Tb_1_spa = Spatial_extraction(Tb_1_spe, 1)

    x_1_GRU = ConvGRU(Ta_1_spa, Tb_1_spa, 1)

    Ta_1_spe_attention = Differential_guided_attention(x_1_GRU, Ta_1_spe)
    Tb_1_spe_attention = Differential_guided_attention(x_1_GRU, Tb_1_spe)

    Ta_2_spe = Spectral_extraction(Ta_1_spe_attention, 2)
    Tb_2_spe = Spectral_extraction(Tb_1_spe_attention, 2)

    Ta_2_spa = Spatial_extraction(Ta_2_spe, 2)
    Tb_2_spa = Spatial_extraction(Tb_2_spe, 2)

    x_2_GRU = ConvGRU(Ta_2_spa, Tb_2_spa, 2)

    Ta_2_spe_attention = Differential_guided_attention(x_2_GRU, Ta_2_spe)
    Tb_2_spe_attention = Differential_guided_attention(x_2_GRU, Tb_2_spe)

    Ta_3_spe = Spectral_extraction(Ta_2_spe_attention, 4)
    Tb_3_spe = Spectral_extraction(Tb_2_spe_attention, 4)

    Ta_3_spa = Spatial_extraction(Ta_3_spe, 4)
    Tb_3_spa = Spatial_extraction(Tb_3_spe, 4)

    x_3_GRU = ConvGRU(Ta_3_spa, Tb_3_spa, 4)

    Ta_3_spe_attention = Differential_guided_attention(x_3_GRU, Ta_3_spe)
    Tb_3_spe_attention = Differential_guided_attention(x_3_GRU, Tb_3_spe)

    Ta_4_spe = Spectral_extraction(Ta_3_spe_attention, 8)
    Tb_4_spe = Spectral_extraction(Tb_3_spe_attention, 8)

    Ta_4_spa = Spatial_extraction(Ta_4_spe, 8)
    Tb_4_spa = Spatial_extraction(Tb_4_spe, 8)

    x_4_GRU = ConvGRU(Ta_4_spa, Tb_4_spa, 8)

    x = kl.concatenate((x_1_GRU, x_2_GRU, x_3_GRU, x_4_GRU), axis=3)

    '''二维卷积神经网络'''
    x = kl.GlobalAvgPool2D()(x)
    x = kl.Dense(1)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('sigmoid')(x)

    model = Model([Ta_0, Tb_0], x)

    return model

if __name__ == '__main__':
    Model = model()
    Model.summary()