## **Project 1 - ECG Time Series**

### **Task**
![alt text](./visualization/MITBIH-classes.png "The 5 different classes of the MIT-BIH data")

### **Data**

- **Arrhythmia Dataset**  
    Number of Samples: 109446  
    Number of Categories: 5  
    Sampling Frequency: 125Hz  
    Data Source: [Physionet's MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/)  
    Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]

- **The PTB Diagnostic ECG Database**  
    Number of Samples: 14552  
    Number of Categories: 2  
    Sampling Frequency: 125Hz  
    Data Source: [Physionet's PTB Diagnostic ECG Database](https://www.physionet.org/physiobank/database/ptbdb/)

Remark: All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 188.

### **Models**
- **Base models**:
    - LSTM:
    - CNN+LSTM:
- **Transfer models**:
- **XGBoosted models**:

### **Results**

Performance on test data set:

|Models          |MIT-BIH<sup>*</sup>               |PTBDB<sup>*</sup>|PTBDB<sup>°</sup>|PTBDB<sup>&dagger;</sup>|
|----------------|:----------------------:|:-----:|:------:|:---:|
|LSTM + Dense    |F1: **0.**<br>Acc: **0.**|F1: **0.787** Acc: **0.776**<br>AUROC: **0.808** AUPRC: **0.934**||
|CNN + LSTM + Dense |F1: **0.973**<br>Acc: **0.971**|F1: **0.951** Acc: **0.951**<br>AUROC: **0.947** AUPRC: **0.982**|F1: **0.990** Acc: **0.990**<br>AUROC: **0.988** AUPRC: **0.996**|F1: **0.994** Acc: **0.994**<br>AUROC: **0.990** AUPRC: **0.996**|
|LSTM + XGB      |F1: **0.**<br>Acc: **0.**|||-
|CNN + LSTM + XGB|F1: **0.**<br>Acc: **0.**|||-

<sup>*</sup> Only trained on this dataset  
<sup>°</sup> Transfer Learning, pre-trained model trained on MIT-BIH, retrained with **frozen** base layers  
<sup>&dagger;</sup> Transfer Learning, pre-trained model trained on MIT-BIH, retrained with **unfrozen** base layers

`python train.py --config config.yaml`
