import os
import argparse
import numpy as np
from data.make_dataset import load_mitbih, load_ptbdb, upsample
from models.models import get_model, get_transfer_model
from keras import optimizers
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, auc, precision_recall_curve
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

    dirName = './results/'+config['experiment_name']
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")

    ### load & preprocess data ###
    print("Load data ...")
    if config['dataset'] == 'mitbih':
        train, test, y_train, y_test = load_mitbih()
    elif config['dataset'] == 'ptbdb':
        train, test, y_train, y_test = load_ptbdb()

    N_CLASSES = len(np.unique(y_train))
    _, data_dist = np.unique(y_train,return_counts=True)
    print(f"Data distribution: {data_dist}")
    data_dim = train.shape
    print(f"Data shape: {data_dim}")
    train, val, y_train, y_val = train_test_split(train, y_train,
                                                test_size=config['val_split_size'],
                                                stratify=y_train)
    if config['upsample']:
        train_up, y_train_up = upsample(train, y_train)

    ### define model ###
    if config['model_id'] == 'transfer-model':
        model = get_transfer_model(N_CLASSES,config['base_model_path'],
                                    config['base_weights_path'],
                                    config['freeze-weights'])
    else:
        model = get_model(config['model_id'],N_CLASSES)
    model.save(f"{dirName}/{config['experiment_name']}.h5")

    model_yaml = model.to_yaml()
    with open(f"{dirName}/{config['experiment_name']}.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    opt = optimizers.Adam(clipnorm=1.)
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['sparse_categorical_accuracy'])

    ### train ###
    model_path = f"{dirName}/{config['experiment_name']}-weights.h5"
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_weights_only=True,save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=config['patience'], verbose=1)
    csv_logger = CSVLogger(f"{dirName}/{config['experiment_name']}.log")
    lr_schuduler = LearningRateScheduler(exp_decay,verbose=1)
    callbacks_list = [checkpoint, early, csv_logger]
    if config['lr_schuduler']:
        callbacks_list.append(lr_schuduler)

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
    pred_test = model.predict(test,verbose=True).argmax(1)
    pred_val = model.predict(val,verbose=True).argmax(1)
    pred_train = model.predict(train,verbose=True).argmax(1)
    try:
        pickle.dump(pred_test,open(f"{dirName}/{config['experiment_name']}-preds.pkl",'wb'))
    except Exception as e:
        print(e)

    print(classification_report(y_test, pred_test))
    print(confusion_matrix(y_test,pred_test))
    report_dict = {}
    if config['dataset'] == 'ptbdb':
        AUROC = roc_auc_score(y_test,pred_test)
        print(f"AUROC: {AUROC}")
        precision, recall, _ = precision_recall_curve(y_test, pred_test)
        AUPRC = auc(recall, precision)
        print(f"AUPRC: {AUPRC}")
        report_dict = {'AUROC':AUROC.item(), 'AUPRC':AUPRC.item()}

    report_dict = {**report_dict,
                    'test-data':classification_report(y_test,pred_test,output_dict=True),
                    'val-data':classification_report(y_val,pred_val,output_dict=True),
                    'train-data':classification_report(y_train,pred_train,output_dict=True)}
    confmat_dict = {'confusion_matrix':confusion_matrix(y_test, pred_test).tolist()}
    res_dict = {**report_dict, **confmat_dict}
    with open(f'{dirName}/eval.yaml', 'w') as file:
        documents = yaml.dump(res_dict, file)

    return 0

def exp_decay(epoch):
   initial_lrate = 1e-3
   k = 0.1
   lrate = initial_lrate * np.exp(-k*epoch)
   return lrate

if __name__ == '__main__':
    main()
