import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_train_images(val_size=0.2):
    label_path = '../data/raw/data/train_labels/*.npy'
    data_path = '../data/raw/data/train_images/*.npy'
    images = glob.glob(data_path)
    labels = glob.glob(label_path)
    X_train, y_train = list(), list()
    for img_path, l_path in zip(images,labels):
        img = np.load(img_path)
        slides, height, width = img.shape
        img = img.reshape((slides,height,width,1))
        X_train.append(img)
        label = np.load(l_path)
        slides, height, width = label.shape
        label = label.reshape((slides,height,width,1))
        y_train.append(label)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=val_size,
                                                      random_state=42)
    X_train = np.concatenate(X_train,axis=0)
    X_val = np.concatenate(X_val,axis=0)
    y_train = np.concatenate(y_train,axis=0)
    y_val = np.concatenate(y_val,axis=0)
    return X_train, X_val, y_train, y_val

def load_test_images(rotated=False):
    # label_path = '../data/raw/data/test_labels/*.npy'
    if rotated:
        data_path = '../data/raw/data/test_images_randomly_rotated/*.npy'
    else:
        data_path = '../data/raw/data/test_images/*.npy'
    images = glob.glob(data_path)
    # labels = glob.glob(label_path)
    X_test, y_test = list(), list()
    for img_path in images:
        img = np.load(img_path)
        slides, height, width = img.shape
        img = img.reshape((slides,height,width,1))
        X_test.append(img)
    X_test = np.concatenate(X_test,axis=0)
    return X_test

def main():
    train, val, y_train, y_val = load_train_images()
    test = load_test_images(True)
    breakpoint()

if __name__ == '__main__':
    main()
