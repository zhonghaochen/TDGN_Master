import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from GRU import ConvGRU2D
from Datapre import patch_size, bands
import tensorflow as tf

def Spectral_extraction(x):
    x = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.ReLU()(x)
    x = kl.Conv2D(filters=32, kernel_size=(3, 3),padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.ReLU()(x)
    return x

def Spatial_extraction(x):
    x = kl.Reshape((1, x.shape[1], x.shape[2], x.shape[3]))(x)
    return x

def ConvGRU(x_1, x_2):
    x_GRU = kl.concatenate((x_1, x_2), axis=1)
    x_GRU = ConvGRU2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(x_GRU)
    x_GRU = ConvGRU2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False)(x_GRU)
    return x_GRU


def Differential_guided_attention(x_GRU, x_spe):
    avg_pool = kl.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x_GRU)
    max_pool = kl.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x_GRU)
    concat = kl.concatenate((avg_pool, max_pool, ), axis=3)
    spatial_feature = kl.Conv2D(filters=1,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                activation='sigmoid',
                                kernel_initializer='he_normal',
                                use_bias=False)(concat)
    spatial_feature = kl.Reshape((spatial_feature.shape[1], spatial_feature.shape[2], 1))(spatial_feature)
    x = kl.multiply([spatial_feature, x_spe])
    x = kl.add([x, x_spe])
    return x


def model():
    Ta_0 = kl.Input(shape=(patch_size, patch_size, bands))
    Tb_0 = kl.Input(shape=(patch_size, patch_size, bands))

    '''光谱提取'''
    Ta_1_spe = Spectral_extraction(Ta_0)
    Tb_1_spe = Spectral_extraction(Tb_0)

    Ta_1_spa = Spatial_extraction(Ta_1_spe)
    Tb_1_spa = Spatial_extraction(Tb_1_spe)

    x_1_GRU = ConvGRU(Ta_1_spa, Tb_1_spa)

    Ta_1_spe_attention = Differential_guided_attention(x_1_GRU, Ta_1_spe)
    Tb_1_spe_attention = Differential_guided_attention(x_1_GRU, Tb_1_spe)

    Ta_2_spe = Spectral_extraction(Ta_1_spe_attention)
    Tb_2_spe = Spectral_extraction(Tb_1_spe_attention)

    Ta_2_spa = Spatial_extraction(Ta_2_spe)
    Tb_2_spa = Spatial_extraction(Tb_2_spe)

    x_2_GRU = ConvGRU(Ta_2_spa, Tb_2_spa)

    Ta_2_spe_attention = Differential_guided_attention(x_2_GRU, Ta_2_spe)
    Tb_2_spe_attention = Differential_guided_attention(x_2_GRU, Tb_2_spe)

    Ta_3_spe = Spectral_extraction(Ta_2_spe_attention)
    Tb_3_spe = Spectral_extraction(Tb_2_spe_attention)

    Ta_3_spa = Spatial_extraction(Ta_3_spe)
    Tb_3_spa = Spatial_extraction(Tb_3_spe)

    x_3_GRU = ConvGRU(Ta_3_spa, Tb_3_spa)

    Ta_3_spe_attention = Differential_guided_attention(x_3_GRU, Ta_3_spe)
    Tb_3_spe_attention = Differential_guided_attention(x_3_GRU, Tb_3_spe)

    Ta_4_spe = Spectral_extraction(Ta_3_spe_attention)
    Tb_4_spe = Spectral_extraction(Tb_3_spe_attention)

    Ta_spe = kl.add([Ta_1_spe, Ta_2_spe, Ta_3_spe, Ta_4_spe])
    Tb_spe = kl.add([Tb_1_spe, Tb_2_spe, Tb_3_spe, Tb_4_spe])

    Ta_4_spa = Spatial_extraction(Ta_spe)
    Tb_4_spa = Spatial_extraction(Tb_spe)

    x_4_GRU = ConvGRU(Ta_4_spa, Tb_4_spa)

    x = kl.concatenate((x_1_GRU, x_2_GRU, x_3_GRU, x_4_GRU), axis=3)

    '''二维卷积神经网络'''
    x = kl.GlobalAvgPool2D()(x)
    x = kl.Dense(2)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('sigmoid')(x)

    model = Model([Ta_0, Tb_0], x)

    return model

if __name__ == '__main__':
    Model = model()
    Model.summary()