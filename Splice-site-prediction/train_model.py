import os
import argparse
import numpy as np
import pickle
import yaml
import time
from collections import defaultdict
import tensorflow as tf

from data.make_dataset import load_celegans, load_human, upsample, downsample
from models.models import get_model
from keras import optimizers
from keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

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
    # set all parameters not explicitly specified to 'False'
    config = defaultdict(lambda: False, config)

    dirName = './results/'+config['experiment_name']
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")

    ### load & preprocess data ###
    print("Load data ...")
    if config['dataset'] == 'celegans':
        data = load_celegans()
    elif config['dataset'] == 'human':
        data = load_human()

    data_dist = np.unique(data['train']['labels'],return_counts=True)[1]/len(data['train']['labels'])
    print(f"Data distribution: {data_dist}")
    data_dim = data['train']['data'].shape
    print(f"Data shape: {data_dim}")
    if config['upsample']:
        data['train']['data'], data['train']['labels'] = upsample(data['train']['data'], data['train']['labels'])
    elif config['downsample']:
        N_per_class = np.unique(data['train']['labels'],return_counts=True)[1].min()
        data['train']['data'], data['train']['labels'] = downsample(data['train']['data'], data['train']['labels'], N_per_class)

    ### define model ###
    model = get_model(config['model_id'],data['seq_length'], config['kernel_size'])
    #model.save(f"{dirName}/{config['experiment_name']}.h5")

    model_yaml = model.to_yaml()
    with open(f"{dirName}/{config['experiment_name']}.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    opt = optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    ### train ###
    model_path = f"{dirName}/{config['experiment_name']}-weights.h5"
    # define callbacks
    checkpoint = callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_weights_only=True,save_best_only=True,verbose=1)
    early = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=config['patience']*2, verbose=1)
    csv_logger = callbacks.CSVLogger(f"{dirName}/{config['experiment_name']}.log")
    terminate_nan = callbacks.TerminateOnNaN()
    redLRonPlateau = callbacks.ReduceLROnPlateau(monitor='val_loss',mode='min',factor=0.5,patience=config['patience'],min_lr=1e-6,verbose=1)
    callbacks_list = [checkpoint, csv_logger, terminate_nan, early]
    if config['reduce_lr_on_plateau']:
        callbacks_list.append(redLRonPlateau)

    if config['class_weights']:
        class_weights = [1., 1000.]
    else:
        class_weights = [1., 1.]

    print('Start training ...')
    start = time.time()
    history = model.fit(data['train']['data'], data['train']['labels'],
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        shuffle=True,
                        class_weight=class_weights,
                        validation_data = (data['val']['data'],data['val']['labels']),
                        callbacks=callbacks_list)
    end = time.time()
    print('Stopped Training.')
    dt = int(end-start)
    print(f"Training took {dt//3600} h {(dt%3600)//60} min {dt%60} sec")

    ### eval ###
    print("Evaluation:")
    model.load_weights(model_path)
    score_test = model.predict(data['test']['data'],verbose=True)
    pred_test = (score_test > 0.5).astype('int32')
    score_val = model.predict(data['val']['data'],verbose=True)
    pred_val = (score_val > 0.5).astype('int32')
    score_train = model.predict(data['train']['data'],verbose=True)
    pred_train = (score_train > 0.5).astype('int32')
    try:
        pickle.dump(score_test,open(f"{dirName}/{config['experiment_name']}-preds.pkl",'wb'))
    except Exception as e:
        print(e)

    y_train, y_val, y_test = data['train']['labels'], data['val']['labels'], data['test']['labels']
    print(metrics.classification_report(y_test, pred_test,digits=3))
    print(metrics.confusion_matrix(y_test,pred_test))
    report_dict = {}
    AUROC = metrics.roc_auc_score(y_test,score_test)
    print(f"AUROC: {AUROC}")
    precision, recall, _ = metrics.precision_recall_curve(y_test, score_test)
    AUPRC = metrics.auc(recall, precision)
    print(f"AUPRC: {AUPRC}")
    report_dict = {'AUROC':AUROC.item(), 'AUPRC':AUPRC.item()}

    report_dict = {**report_dict,
                    'test-data':metrics.classification_report(y_test,pred_test,output_dict=True),
                    'val-data':metrics.classification_report(y_val,pred_val,output_dict=True),
                    'train-data':metrics.classification_report(y_train,pred_train,output_dict=True)}
    confmat_dict = {'confusion_matrix':metrics.confusion_matrix(y_test, pred_test).tolist()}
    res_dict = {**report_dict, **confmat_dict}
    with open(f'{dirName}/eval.yaml', 'w') as file:
        documents = yaml.dump(res_dict, file)

    return 0

if __name__ == '__main__':
    main()
