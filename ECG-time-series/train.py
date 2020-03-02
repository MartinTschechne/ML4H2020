import argparse
import numpy as np
from data.make_dataset import load_mitbih, load_ptbdb, upsample
from models.models import get_model
from keras import optimizers
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import yaml

def main():
    ### load config ###
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Specify path to yaml config file.')
    args = parser.parse_args()
    try:
        with open(args.config,'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    ### load & preprocess data ###
    print("Load data ...")
    if config['dataset'] == 'mitbih':
        train, test, y_train, y_test = load_mitbih()
    elif conig['dataset'] == 'ptbdb':
        train, test, y_train, y_test = load_ptbdb()

    N_CLASSES = len(np.unique(y_train))
    data_dim = train.shape
    train, val, y_train, y_val = train_test_split(train, y_train,
                                                test_size=config['val_split_size'],
                                                stratify=y_train)
    if config['upsample']:
        train_up, y_train_up = upsample(train, y_train)

    ### define model ###
    model = get_model(config['model_id'],N_CLASSES)
    model.save(f"./{config['experiment_name']}.h5")

    model_yaml = model.to_yaml()
    with open(f"{config['experiment_name']}.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    opt = optimizers.Adam(clipnorm=1.)
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    ### train ###
    model_path = f"{config['experiment_name']}-weights.h5"
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_weights_only=True,save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
    csv_logger = CSVLogger(f"{config['experiment_name']}.log")
    callbacks_list = [checkpoint, early, csv_logger]

    ### for debugging only
    if config['debug']:
        train_up, _, y_train_up, _ = train_test_split(train_up, y_train_up,train_size=100)

    print('Start training ...')
    history = model.fit(train_up, y_train_up,
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        shuffle=True,
                        validation_data = (val,y_val),
                        callbacks=callbacks_list)
    print('Stopped Training.')

    ### eval ###
    print("Evaluation:")
    model.load_weights(model_path)
    pred_test = model.predict(test)
    try:
        pickle.dump(pred_test,open(f"{config['experiment_name']}-preds.pkl",'wb'))
    except Exception as e:
        print(e)
    pred_test = np.argmax(pred_test, axis=1)

    print(classification_report(y_test, pred_test))

    print(confusion_matrix(y_test,pred_test))

    return 0

if __name__ == '__main__':
    main()
