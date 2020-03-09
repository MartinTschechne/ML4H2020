## **Project 1 - ECG Time Series**

### **Task**
<p align="center"><img src="./visualization/MITBIH-classes.png" alt="The 5 different classes of the MIT-BIH data" width="500"></p>

### **Data**

- **Arrhythmia Dataset**  
    Number of Samples: 109446  
    Number of Categories: 5  
    Sampling Frequency: 125Hz  
    Data Source: [Physionet's MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/)  
    Classes: ['N' (Normal):                        0,
              'S' (Supraventricular ectopic beat): 1,
              'V' (Ventricular ectopic beat):      2,
              'F' (Fusion beat):                   3,
              'Q' (Unknown beat):                  4]

- **The PTB Diagnostic ECG Database**  
    Number of Samples: 14552  
    Number of Categories: 2  
    Sampling Frequency: 125Hz  
    Data Source: [Physionet's PTB Diagnostic ECG Database](https://www.physionet.org/physiobank/database/ptbdb/)

Remark: All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 188.

### **Models**
- **Base Models**:
    - LSTM + FC:
    <p align="center"><img src="./visualization/lstm.png" alt="The 5 different classes of the MIT-BIH data" width="200"></p>
    - CNN + LSTM + FC: A combination of 1D max-pooled convolutional layers with one succeeding LSTM cell, followed by relu-activated fully connected layers and one softmax output layer.  
    <p align="center"><img src="./visualization/cnn-lstm.png" alt="The 5 different classes of the MIT-BIH data" width="200"></p>

    We refer to all layers before the fully connected layers as *base layers* or *feature extractors*.
- **Transfer Learning Models**:
    The idea of transfer learning is to use a pre-trained model on another (preferably large) dataset solving essentially the same task we like to do with a different (usually smaller) dataset. In the case at hand of ECG classification, we use the large MIT-BIH dataset to train a model. Then we used that pre-trained model and retrained it on the much smaller PTBDB dataset. We further differentiate between two methods of training here: *frozen* base layers and *unfrozen* base layers. Frozen base layers refers to not updating the base layers of the pre-trained model, whereas in the unfrozen method we train all layers.
- **XGBoosted Models**:
    *Gradient Boosting* has proven many times in the past to be a very effective algorithm for a large variety of machine learning task by winning multiple Kaggle competitions. In this study we compare the performance of fully connected layers and gradient boosting by using the pre-trained base layers are feature extractors and replace the fully connected layers with a gradient boosting model, XGBoost[3].  
    Further we also examine the transfer learning performance of XGBoost by using a feature extractor trained on the MIT-BIH dataset and training the XGBoost model on the so extracted features from the PTBDB dataset.  

### **Results**

MIT-BIH No Information Rate: **0.828**  
PTBDB No Information Rate: **0.722**  
Performance on test data set:

|Models          |MIT-BIH<sup>*</sup>|PTBDB<sup>*</sup>|PTBDB<sup>°</sup>|PTBDB<sup>&dagger;</sup>|
|----------------|:----------------------:|:-----:|:------:|:---:|
|LSTM + FC    |F1: **0.184**<br>Acc: **0.823**|F1: **0.787** Acc: **0.776**<br>AUROC: **0.808** AUPRC: **0.934**|F1: **0.419** Acc: **0.722**<br>AUROC: **0.5** AUPRC: **0.861**|F1: **0.371** Acc: **0.565**<br>AUROC: **0.397** AUPRC: **0.805**|
|CNN + LSTM + FC |F1: **0.868**<br>Acc: **0.971**|F1: **0.940** Acc: **0.951**<br>AUROC: **0.947** AUPRC: **0.982**|F1: **0.988** Acc: **0.990**<br>AUROC: **0.988** AUPRC: **0.996**|F1: **0.992** Acc: **0.994**<br>AUROC: **0.990** AUPRC: **0.996**|
|LSTM + XGB<sup>&Dagger;</sup> |F1: **0.875**<br>Acc: **0.976**|F1: **0.971** Acc: **0.977**<br>AUROC: **0.968** AUPRC: **0.988**|F1: **0.963** Acc: **0.970**<br>AUROC: **0.955** AUPRC: **0.983**| - |
|CNN + LSTM + XGB<sup>&Dagger;</sup>|F1: **0.916**<br>Acc: **0.985**|F1: **0.983** Acc: **0.986**<br>AUROC: **0.980** AUPRC: **0.993**|F1: **0.981** Acc: **0.990**<br>AUROC: **0.977** AUPRC: **0.991**|-|
|XGB|F1: **0.896**<br>Acc: **0.979**|F1: **0.970** Acc: **0.976**<br>AUROC: **0.966** AUPRC: **0.987**| - | - |
|Kachuee, et al.[1]|Acc: **0.934**| - |F1: **0.951**<br>Acc: **0.959**| - |
|Baseline[2]|F1: **0.915**<br>Acc: **0.985**|F1: **0.988**<br>Acc: **0.983**|F1: **0.969**<br>Acc: **0.956**|F1: **0.994**<br>Acc: **0.992**|

<sup>*</sup> Only trained on this dataset  
<sup>°</sup> Transfer Learning, pre-trained model trained on MIT-BIH, retrained with **frozen** base layers  
<sup>&dagger;</sup> Transfer Learning, pre-trained model trained on MIT-BIH, retrained with **unfrozen** base layers  
<sup>&Dagger;</sup> Base layers always frozen to train XGBoost

### Embedding Visualizations

||t-SNE|UMAP|PCA|  
|:---:|:---:|:---:|:---:|  
|**MITBIH**|<img src="visualization/mitbih-tsne-50.png?" width="250">|<img src="visualization/mitbih-umap.png?" width="250">|<img src="visualization/mitbih-pca.png?" width="250">|
|**PTBDB**|<img src="visualization/ptbdb-tsne-50.png?" width="250">|<img src="visualization/ptbdb-umap.png?" width="250">|<img src="visualization/ptbdb-pca.png?" width="250">|

### Reproducibility

To reproduce the results, download the zipped data form the sources mentioned above. Create the folders `data` and `data/raw` inside the project folder. Extract the zip-file inside `data/raw`.

 select the configuration file corresponding the the model configuration in the table above from the `config-file` directory and run

`python train.py --config ./config-files/config.yaml`

for base and transfer models,  

`python train_base_xgb.py --config ./config-files/gxb-config.yaml`  

XGBoosted models and

`python train_xgb.py --config ./config-files/gxb-config.yaml`

to reproduce reference results of pure XGBoost models.

Performance metrics for each model as well as weights, architectures and test set predictions are saved in the `results` folder in a single folder for each model.

### References

[1] Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification: A Deep Transferable Representation." [arXiv preprint arXiv:1805.00794 (2018) ](https://arxiv.org/abs/1805.00794).

[2] CVxTz's GitHub implementation: ECG_Heartbeat_Classification ([link](https://github.com/CVxTz/ECG_Heartbeat_Classification))

[3] XGBoost ([https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost))

[4] McInnes, L, Healy, J. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." [ArXiv e-prints 1802.03426, 2018](https://arxiv.org/abs/1802.03426).
