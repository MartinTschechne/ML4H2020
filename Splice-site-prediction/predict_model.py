import os
import argparse
import pickle
import yaml
import time
import numpy as np
from collections import defaultdict

from data.make_dataset import load_human_final_test
from models.models import get_model
from sklearn import metrics

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
    for k,v in config.items():
        print(k,':',v)

    dirName = './results/'+config['experiment_name']

    ### load & preprocess data ###
    print("Load data ...")
    if config['dataset'] == 'human':
        data = load_human_final_test()
    else:
        print("No test data set found.")
        exit(-1)

    ### load model ###
    model = get_model(config['model_id'],data['seq_length'], config['kernel_size'])
    model_path = f"{dirName}/{config['experiment_name']}-weights.h5"
    model.load_weights(model_path)

    ### start inference ###
    print('Start predicting ...')
    start = time.time()
    score_test = model.predict(data['test']['data'],verbose=True)
    score_hidden = model.predict(data['hidden']['data'],verbose=True)
    end = time.time()
    print('Prediction done.')
    dt = int(end-start)
    print(f"Inference took {dt//3600} h {(dt%3600)//60} min {dt%60} sec")
    try:
        pickle.dump(score_test,open(f"{dirName}/{config['experiment_name']}-test-preds.pkl",'wb'))
        pickle.dump(score_hidden,open(f"{dirName}/{config['experiment_name']}-hidden-preds.pkl",'wb'))
    except Exception as e:
        print(e)

    ### evaluation ###
    y_test = data['test']['labels']
    pred_test = (score_test > 0.5).astype('int32')
    pred_hidden = (score_hidden > 0.5).astype('int32')
    with open(f'{dirName}/results.npy','wb') as f:
        np.save(f, pred_hidden)
    print("Test scores:")
    print(metrics.classification_report(y_test, pred_test,digits=3))
    print(metrics.confusion_matrix(y_test,pred_test))
    AUROC = metrics.roc_auc_score(y_test,score_test)
    print(f"AUROC: {AUROC}")
    precision, recall, _ = metrics.precision_recall_curve(y_test, score_test)
    AUPRC = metrics.auc(recall, precision)
    print(f"AUPRC: {AUPRC}")

    report_dict = {'test-data':metrics.classification_report(y_test,pred_test,output_dict=True),
                    'AUROC':AUROC.item(), 'AUPRC':AUPRC.item()}
    confmat_dict = {'test-confusion_matrix':metrics.confusion_matrix(y_test, pred_test).tolist()}
    res_dict = {**report_dict, **confmat_dict}
    with open(f'{dirName}/test-eval.yaml', 'w') as file:
        documents = yaml.dump(res_dict, file)

    return 0

if __name__ == '__main__':
    main()
