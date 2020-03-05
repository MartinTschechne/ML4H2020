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
- **Base models**:
    - LSTM + Dense:
    - CNN + LSTM + Dense:
- **Transfer models**:
- **XGBoosted models**:

### **Results**

MIT-BIH No Information Rate: **0.828**  
PTBDB No Information Rate: **0.722**  
Performance on test data set:

|Models          |MIT-BIH<sup>*</sup>|PTBDB<sup>*</sup>|PTBDB<sup>°</sup>|PTBDB<sup>&dagger;</sup>|
|----------------|:----------------------:|:-----:|:------:|:---:|
|LSTM + Dense    |F1: **0.184**<br>Acc: **0.823**|F1: **0.787** Acc: **0.776**<br>AUROC: **0.808** AUPRC: **0.934**|F1: **0.419** Acc: **0.722**<br>AUROC: **0.5** AUPRC: **0.861**||
|CNN + LSTM + Dense |F1: **0.868**<br>Acc: **0.971**|F1: **0.940** Acc: **0.951**<br>AUROC: **0.947** AUPRC: **0.982**|F1: **0.988** Acc: **0.990**<br>AUROC: **0.988** AUPRC: **0.996**|F1: **0.992** Acc: **0.994**<br>AUROC: **0.990** AUPRC: **0.996**|
|LSTM + XGB<sup>&Dagger;</sup> |F1: **0.875**<br>Acc: **0.976**|F1: **0.971** Acc: **0.977**<br>AUROC: **0.968** AUPRC: **0.988**|F1: **0.963** Acc: **0.970**<br>AUROC: **0.955** AUPRC: **0.983**| - |
|CNN + LSTM + XGB<sup>&Dagger;</sup>|F1: **0.916**<br>Acc: **0.985**|F1: **0.983** Acc: **0.986**<br>AUROC: **0.980** AUPRC: **0.993**|F1: **0.981** Acc: **0.990**<br>AUROC: **0.977** AUPRC: **0.991**|-|
|Kachuee, et al.[1]|Acc: **0.934**| - |F1: **0.951**<br>Acc: **0.959**| - |
|Baseline[2]|F1: **0.915**<br>Acc: **0.985**|F1: **0.988**<br>Acc: **0.983**|F1: **0.969**<br>Acc: **0.956**|F1: **0.994**<br>Acc: **0.992**|

<sup>*</sup> Only trained on this dataset  
<sup>°</sup> Transfer Learning, pre-trained model trained on MIT-BIH, retrained with **frozen** base layers  
<sup>&dagger;</sup> Transfer Learning, pre-trained model trained on MIT-BIH, retrained with **unfrozen** base layers  
<sup>&Dagger;</sup> Base layers always frozen to train XGBoost

`python train.py --config config.yaml`

### References

[1] Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification: A Deep Transferable Representation." [arXiv preprint arXiv:1805.00794 (2018) ](https://arxiv.org/abs/1805.00794).

[2] CVxTz's GitHub implementation: ECG_Heartbeat_Classification ([link](https://github.com/CVxTz/ECG_Heartbeat_Classification))
