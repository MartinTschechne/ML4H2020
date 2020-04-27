import os
import argparse
import numpy as np
import pickle
import yaml
import time

from keras import Model

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             jaccard_score)

from models.unet import UNET
from data.make_dataset import load_test_images

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
    print("Configuration:")
    for k,v in config.items():
        print(k,':',v)

    dirName = './results/'+config['experiment_name']
    test, y_test = load_test_images()
    test_rotated, y_test_rotated = load_test_images(rotated=True)

    ### load model ###
    N_CLASSES = 3
    INPUT_SHAPE = (256,256,1)

    unet = UNET(input_shape = INPUT_SHAPE,
                num_classes = N_CLASSES,
                seed = SEED,
                **config['model'])

    ### evaluation ###
    print("Evaluation:")
    model_path = f"{dirName}/{config['experiment_name']}-weights.h5"
    unet.model.load_weights(model_path)
    pred_test = unet.model.predict(test,verbose=True).argmax(-1)
    pred_test_rotated = unet.model.predict(test_rotated,verbose=True).argmax(-1)
    try:
        pickle.dump(pred_test,open(f"{dirName}/{config['experiment_name']}-test-preds.pkl",'wb'))
        pickle.dump(pred_test_rotated,open(f"{dirName}/{config['experiment_name']}-test-rot-preds.pkl",'wb'))
    except Exception as e:
        print(e)

    pred_test = pred_test.flatten()
    pred_test_rotated = pred_test_rotated.flatten()
    y_test = y_test.flatten()
    y_test_rotated = y_test_rotated.flatten()

    print("Test scores:")
    print(classification_report(y_test, pred_test))
    print(confusion_matrix(y_test,pred_test))
    print(jaccard_score(y_test,pred_test,average='macro'))

    print("Rotated test scores:")
    print(classification_report(y_test_rotated, pred_test_rotated))
    print(confusion_matrix(y_test_rotated,pred_test_rotated))
    print(jaccard_score(y_test_rotated,pred_test_rotated,average='macro'))

    report_dict = {'test-data':classification_report(y_test,pred_test,output_dict=True),
                    'test-rot-data':classification_report(y_test_rotated,pred_test_rotated,output_dict=True),
                    'test-jaccard-score': jaccard_score(y_test,pred_test,average='macro').item(),
                    'test-rot-jaccard-score': jaccard_score(y_test_rotated,pred_test_rotated,average='macro').item()}
    confmat_dict = {'test-confusion_matrix':confusion_matrix(y_test, pred_test).tolist(),
                    'test-rot-confusion_matrix':confusion_matrix(y_test_rotated, pred_test_rotated).tolist()}
    res_dict = {**report_dict, **confmat_dict}
    with open(f'{dirName}/test-eval.yaml', 'w') as file:
        documents = yaml.dump(res_dict, file)
    return 0

if __name__ == '__main__':
    main()
