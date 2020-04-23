import os
import argparse
import numpy as np
import pickle
import yaml

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             jaccard_score)

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler

from models.unet import UNET, Jaccard_XEntropy_Loss, Focal_Loss
from data.make_dataset import load_train_images

from keras_contrib.losses.jaccard import jaccard_distance

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
    print("Configuration:")
    for k,v in config.items():
        print(k,':',v)

    dirName = './results/'+config['experiment_name']
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")

    ### load & preprocess data ###
    SEED = config['seed']
    train, val, y_train, y_val = load_train_images()
    if config['augmentation']:
        data_gen_args = dict(rotation_range=config['rot_range'],
                             zoom_range=config['zoom_range'],
                             vertical_flip=config['vertical_flip'],
                             fill_mode='reflect')
    else:
        data_gen_args = dict(rotation_range=0.,
                             zoom_range=0.,
                             fill_mode='reflect')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow(train,batch_size=config['batch_size'],seed=SEED)
    mask_generator = mask_datagen.flow(to_categorical(y_train,3),batch_size=config['batch_size'],seed=SEED)
    train_generator = zip(image_generator, mask_generator)

    ### define model ###
    N_CLASSES = 3
    INPUT_SHAPE = (256,256,1)

    unet = UNET(INPUT_SHAPE, N_CLASSES,
                config['filter_list'], config['kernel_size'], config['batch_norm'])

    # unet.model.save(f"{dirName}/{config['experiment_name']}.h5")

    model_yaml = unet.model.to_yaml()
    with open(f"{dirName}/{config['experiment_name']}.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    unet.model.compile(loss=get_loss(config),
                  optimizer=get_optimizer(config),
                  metrics=[jaccard_distance, 'accuracy'])

    ### train ###
    model_path = f"{dirName}/{config['experiment_name']}-weights.h5"
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_weights_only=True,save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=config['patience'], verbose=1)
    csv_logger = CSVLogger(f"{dirName}/{config['experiment_name']}.log")
    lr_scheduler = LearningRateScheduler(exp_decay,verbose=1)
    callbacks_list = [checkpoint, early, csv_logger]
    if config['lr_scheduler']:
        callbacks_list.append(lr_scheduler)

    if config['class_weights'] == 'balanced':
        class_weights = compute_class_weight('balanced',np.unique(y_train),y_train.flatten())
    elif config['class_weights'] == 'weighted':
        class_weights = [1., 100., 10.]
    else:
        class_weights = [1., 1., 1.]

    print('Start training ...')
    history = unet.model.fit_generator(train_generator,
                        steps_per_epoch=len(train)//config['batch_size'],
                        epochs=config['epochs'],
                        class_weight=class_weights,
                        shuffle=True,
                        validation_data = (val,to_categorical(y_val,3)),
                        callbacks=callbacks_list,
                        use_multiprocessing=True)
    print('Stopped Training.')

    ### evaluation ###
    print("Evaluation:")
    unet.model.load_weights(model_path)
    pred_val = unet.model.predict(val,verbose=True).argmax(-1)
    pred_train = unet.model.predict(train,verbose=True).argmax(-1)
    try:
        pickle.dump(pred_val,open(f"{dirName}/{config['experiment_name']}-preds.pkl",'wb'))
        # pickle.dump(y_val,open(f"{dirName}/{config['experiment_name']}-labels.pkl",'wb'))
    except Exception as e:
        print(e)

    pred_val = pred_val.flatten()
    pred_train = pred_train.flatten()
    y_val = y_val.flatten()
    y_train = y_train.flatten()

    print(classification_report(y_val, pred_val))
    print(confusion_matrix(y_val,pred_val))
    print(jaccard_score(y_val,pred_val,average='micro'))

    report_dict = {'val-data':classification_report(y_val,pred_val,output_dict=True),
                    'train-data':classification_report(y_train,pred_train,output_dict=True),
                    'val-jaccard-score': jaccard_score(y_val,pred_val,average='micro').item(),
                    'train-jaccard-score': jaccard_score(y_train,pred_train,average='micro').item()}
    confmat_dict = {'val-confusion_matrix':confusion_matrix(y_val, pred_val).tolist()}
    res_dict = {**report_dict, **confmat_dict}
    with open(f'{dirName}/eval.yaml', 'w') as file:
        documents = yaml.dump(res_dict, file)
    return 0

def get_loss(config):
    if config['loss'] == 'cross-entropy':
        return 'categorical_crossentropy'
    elif config['loss'] == 'jaccard':
        return jaccard_distance
    elif config['loss'] == 'jaccard-xentropy':
        return Jaccard_XEntropy_Loss(config['alpha'])
    elif config['loss'] == 'focal':
        return Focal_Loss()

def get_optimizer(config):
    if config['optimizer'] == 'adam':
        if config['lr'] == 'default':
            return optimizers.Adam()
        else:
            return optimizers.Adam(lr=config['lr'])
    if config['optimizer'] == 'sgd':
        return optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

def exp_decay(epoch):
   initial_lrate = 1e-3
   k = 0.1
   lrate = initial_lrate * np.exp(-k*epoch)
   return lrate

if __name__ == '__main__':
    main()
