import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_mitbih():
    df_train = pd.read_csv("../data/raw/mitbih_train.csv", header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv("../data/raw/mitbih_test.csv", header=None)

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    return X, X_test, Y, Y_test

def load_ptbdb():
    df_1 = pd.read_csv("../data/raw/ptbdb_normal.csv", header=None)
    df_2 = pd.read_csv("../data/raw/ptbdb_abnormal.csv", header=None)
    df = pd.concat([df_1, df_2])

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    return X, X_test, Y, Y_test

def upsample(data, labels, seed=42):
    '''
    Upsample minority classes up to the majority class.
    Returned data is NOT shuffled.
    '''
    CLASSES, N_SAMPLES = np.unique(labels,return_counts=True)
    MAX = np.max(N_SAMPLES)
    data_upsampled = []
    labels_upsampled = []
    for c, n in zip(CLASSES, N_SAMPLES):
        data_sub = data[labels==c]
        data_sampled = resample(data_sub,
                                replace=True,
                                n_samples=MAX - n,
                                random_state=seed)
        data_upsampled.append(np.vstack([data_sub, data_sampled]))
        labels_upsampled.append(np.ones(MAX,np.int8)*c)

    data_upsampled = np.vstack(data_upsampled)
    labels_upsampled = np.hstack(labels_upsampled)
    return data_upsampled, labels_upsampled

def main():
    print(f"Test upsampling for MITBIH:")
    train, test, y_train, y_test = load_mitbih()
    print(f"Before upsampling:")
    CLASSES, N_SAMPLES = np.unique(y_train,return_counts=True)
    print(f"Classes: {CLASSES},\nsamples per class: {N_SAMPLES}")
    train_upsampled, y_upsampled = upsample(train, y_train)
    print(f"After upsampling:")
    _, NEW_SAMPLES = np.unique(y_upsampled,return_counts=True)
    print(f"Samples per class: {NEW_SAMPLES}\n\n")

    print(f"Test upsampling for PTBDB:")
    train, test, y_train, y_test = load_ptbdb()
    print(f"Before upsampling:")
    CLASSES, N_SAMPLES = np.unique(y_train,return_counts=True)
    print(f"Classes: {CLASSES},\nsamples per class: {N_SAMPLES}")
    train_upsampled, y_upsampled = upsample(train, y_train)
    print(f"After upsampling:")
    _, NEW_SAMPLES = np.unique(y_upsampled,return_counts=True)
    print(f"Samples per class: {NEW_SAMPLES}")
    return 0

if __name__ == '__main__':
    main()
