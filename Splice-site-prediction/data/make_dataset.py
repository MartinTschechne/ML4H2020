import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def downsample(data, labels, N_per_class, seed=42):
    '''
    Upsample minority classes up to the majority class.
    Returned data is NOT shuffled.
    '''
    CLASSES, N_SAMPLES = np.unique(labels,return_counts=True)
    data_downsampled = []
    labels_downsampled = []
    for c, n in zip(CLASSES, N_SAMPLES):
        data_sub = data[labels==c]
        data_sampled = resample(data_sub,
                                replace=False,
                                n_samples=N_per_class,
                                random_state=seed)
        data_downsampled.append(data_sampled)
        labels_downsampled.append(np.ones(N_per_class,np.int8)*c)

    data_downsampled = np.vstack(data_downsampled)
    labels_downsampled = np.hstack(labels_downsampled)
    return data_downsampled, labels_downsampled

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

def encode_seq(seq):
    '''Endode single sequence into numeric representation.'''
    char2num = {'A':0, 'C': 1, 'G': 2, 'T': 3}
    return [char2num[c] for c in seq]

def preprocess_sequences(df):
    '''Process raw dataframe and return preprocessed data and labels.'''
    data = df.sequences.values
    data = [encode_seq(d) for d in data]
    data = np.array(data)
    try:
        data = data.reshape((data.shape[0],data.shape[1],1))
        labels = df.labels.values
        labels[labels == -1] = 0
    except:
        labels = 0
    seq_length = len(data[0])
    return data/3, np.int8(labels), seq_length

def load_human():
    '''Load train and vaildation data and split further split into train, val, test.'''
    train_df = pd.read_csv('../data/raw/exercise_data/human_dna_train_split.csv')
    train, y_train, seq_length = preprocess_sequences(train_df)
    train, val, y_train, y_val = train_test_split(train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    train_data = {'data':train, 'labels':y_train}
    val_data = {'data':val, 'labels':y_val}
    test_df = pd.read_csv('../data/raw/exercise_data/human_dna_validation_split.csv')
    tes, y_test, seq_length = preprocess_sequences(test_df)
    test_data = {'data':tes, 'labels':y_test}
    return {'train':train_data, 'val':val_data, 'test':test_data, 'seq_length':seq_length}

def load_human_final_test():
    '''Load test data and hidden test data for final prediction.'''
    hidden_df = pd.read_csv('../data/raw/exercise_data/human_dna_test_hidden_split.csv')
    hidden, _, seq_length = preprocess_sequences(hidden_df)
    hidden_data = {'data':hidden}
    test_df = pd.read_csv('../data/raw/exercise_data/human_dna_test_split.csv')
    test, y_test, seq_length = preprocess_sequences(test_df)
    test_data = {'data':test, 'labels':y_test}
    return {'test':test_data, 'hidden':hidden_data, 'seq_length':seq_length}


def load_celegans():
    '''Load C.Elegans DNA data.'''
    df = pd.read_csv('../data/raw/exercise_data/C_elegans_acc_seq.csv',header=None,names=['labels','sequences'])
    data, labels, seq_length = preprocess_sequences(df)
    train, test, y_train, y_test = train_test_split(data, labels, test_size=0.1, stratify=labels, random_state=42)
    train, val, y_train, y_val = train_test_split(train, y_train, test_size=0.1/0.9, stratify=y_train, random_state=42)
    train_data = {'data':train, 'labels':y_train}
    val_data = {'data':val, 'labels':y_val}
    test_data = {'data':test, 'labels':y_test}
    return {'train':train_data, 'val':val_data , 'test':test_data ,'seq_length':seq_length}

def main():
    return 0

if __name__ == '__main__':
    main()
