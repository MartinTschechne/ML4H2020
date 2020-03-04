import os
import argparse
import numpy as np
from data.make_dataset import load_mitbih, load_ptbdb, upsample
from models.models import get_feature_extractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, auc, precision_recall_curve
from xgboost import XGBClassifier
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

    dirName = config['experiment_name']
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")

    ### load & preprocess data ###
    print("Load data ...")
    if config['dataset'] == 'mitbih':
        train, test, y_train, y_test = load_mitbih()
        objective = 'multi:softmax'
        eval_metric = ['merror', 'mlogloss']
    elif config['dataset'] == 'ptbdb':
        train, test, y_train, y_test = load_ptbdb()
        objective = 'binary:logistic'
        eval_metric = ['logloss', 'aucpr', 'auc']
    N_CLASSES = len(np.unique(y_train))
    train, val, y_train, y_val = train_test_split(train, y_train,
                                                test_size=config['val_split_size'],
                                                stratify=y_train)

    ### define model ###
    feature_extractor = get_feature_extractor(config['base_model_path'],
                          config['base_weights_path'])

    xgb_model = XGBClassifier(max_depth=10,
                              n_estimators=256,
                              objective=objective,
                              eval_metric=eval_metric,
                              learning_rate = 0.1,
                              nthread=4,
                              random_state=42)

    print('Start training ...')
    train_features = feature_extractor.predict(train,verbose=True)
    val_features = feature_extractor.predict(val,verbose=True)
    xgb_model.fit(train_features,y_train,
                  eval_set=[(val_features,y_val)],
                  early_stopping_rounds=3,
                  verbose=True)
    print('Stopped Training.')
    model_path = f"./{dirName}/{config['experiment_name']}-xgb.json"
    xgb_model.save_model(model_path)

    ### eval ###
    test_features = feature_extractor.predict(test,verbose=True)
    pred_test = xgb_model.predict(test_features)
    pred_val = xgb_model.predict(val_features)
    pred_train = xgb_model.predict(train_features)
    try:
        pickle.dump(pred_test,open(f"./{dirName}/{config['experiment_name']}-preds.pkl",'wb'))
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
    with open(f'./{dirName}/eval.yaml', 'w') as file:
        documents = yaml.dump(res_dict, file)

    return 0



if __name__ == '__main__':
    main()
