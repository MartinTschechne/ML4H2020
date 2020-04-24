import math
from keras.losses import categorical_crossentropy
from keras.backend import reshape
from keras_contrib.losses.jaccard import jaccard_distance
from keras.models import Model
from keras import layers
import keras.backend as K

class UNET():

    def __init__(self,
                 input_shape,
                 num_classes,
                 filter_list,
                 kernel_size,
                 seed,
                 activation = 'relu',
                 dropout = False,
                 dropout_rate = 0.3,
                 batch_norm = False,
                 initializer = 'glorot_uniform'):
        '''
        UNET model
        '''
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filter_list = filter_list
        self.kernel_size = kernel_size
        self.seed = seed
        self.activation = activation
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.num_passes = len(self.filter_list)
        self.batch_norm = batch_norm
        self.initializer = initializer
        self.copies = list()

        # model
        input = layers.Input(shape=self.input_shape)
        # down
        x = layers.BatchNormalization()(input) # learn normalization
        if self.dropout:
            x = layers.Dropout(rate=0.2,seed=self.seed)(x) # keep 80% of input
        for f in self.filter_list:
            x, copy = self.down_pass(x,f)
            self.copies.append(copy)

        x = self.bottom(x,self.filter_list[-1]*2)
        self.copies.reverse()
        self.filter_list.reverse()
        # up
        for i, (f, copy) in enumerate(zip(self.filter_list,self.copies),1):
            x = self.up_pass(x,copy,f,last = i == self.num_passes)

        predictions = layers.Conv2D(filters=self.num_classes,
                             kernel_size=(1,1),
                             activation='softmax',
                             padding='same')(x)
        self.model = Model(inputs=input,outputs=predictions)

    def down_pass(self,input,n_filters):
        '''Down pass'''
        output = layers.Conv2D(n_filters,self.kernel_size,padding='same')(input)
        output = layers.Activation(self.activation)(output)
        output = layers.Conv2D(n_filters,self.kernel_size,padding='same')(output)
        output = layers.Activation(self.activation)(output)
        if self.batch_norm:
            output = layers.BatchNormalization()(output)
        copy = output
        output = layers.MaxPooling2D()(output)
        if self.dropout:
            output = layers.Dropout(rate=self.dropout_rate,seed=self.seed)(output)
        return output, copy

    def up_pass(self,input,copy,n_filters, last=False):
        '''Up pass'''
        output = layers.Concatenate(axis=-1)([input,copy])
        output = layers.Conv2D(n_filters,self.kernel_size,padding='same')(output)
        output = layers.Activation(self.activation)(output)
        output = layers.Conv2D(n_filters,self.kernel_size,padding='same')(output)
        output = layers.Activation(self.activation)(output)
        if self.batch_norm:
            output = layers.BatchNormalization()(output)
        if not last:
            output = layers.Conv2DTranspose(n_filters//2,self.kernel_size,strides=(2,2),padding='same')(output)
            if self.dropout:
                output = layers.Dropout(rate=self.dropout_rate,seed=self.seed)(output)
        return output

    def bottom(self,input,n_filters):
        output = layers.Conv2D(n_filters,self.kernel_size,padding='same')(input)
        output = layers.Activation(self.activation)(output)
        output = layers.Conv2D(n_filters,self.kernel_size,padding='same')(output)
        output = layers.Activation(self.activation)(output)
        if self.batch_norm:
            output = layers.BatchNormalization()(output)
        output = layers.Conv2DTranspose(n_filters//2,self.kernel_size,strides=(2,2),padding='same')(output)
        if self.dropout:
            output = layers.Dropout(rate=self.dropout_rate,seed=self.seed)(output)
        return output

def Jaccard_XEntropy_Loss(alpha=0.5):
    def JX_Loss(y_true,y_pred):
        xe_loss = categorical_crossentropy(y_true,y_pred)
        jac_dis = jaccard_distance(y_true,y_pred)
        return alpha*jac_dis + (1.-alpha)*xe_loss
    return JX_Loss

def Focal_Loss(gamma=2, alpha=.25):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss, axis=1)

    return categorical_focal_loss_fixed


def main():
    unet = UNET(input_shape = (256,256,1),
                num_classes = 3,
                filter_list = [64,128,256,512],
                kernel_size = 3,
                seed = 42,
                activation = 'relu',
                dropout = True,
                dropout_rate = 0.3,
                batch_norm = True)
    unet.model.summary()

if __name__ == '__main__':
    main()
