import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam
from models.unet import UNET, make_Jaccard_XEntropy_Loss
from data.make_dataset import load_train_images
from keras_contrib.losses.jaccard import jaccard_distance

def main():
    pass

if __name__ == '__main__':
    main()
