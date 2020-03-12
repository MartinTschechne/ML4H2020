ML4H2020
==============================

Repository for projects of the course "Machine Learning for Health Care".

**Authors** :  
Han Bai  
Nora Moser  
Martin Tschechne (martints@ethz.ch)


## Project 1 - ECG Time Series
Classifying ECG signals of the [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/physiobank/database/mitdb/) and the [PTB Diagnostic ECG Database](https://www.physionet.org/physiobank/database/ptbdb/) by Recurrent Neural Networks and make use of Transfer Learning techniques in order to improve predictive performance. Finally compare models performance to the results obtained by Kachuee, et al.[1]

**Results**

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

**Visualization of learned embeddings**
<center>

||t-SNE|UMAP|PCA|  
|:---:|:---:|:---:|:---:|  
|**MIT-BIH**|<img src="ECG-time-series/visualization/mitbih-tsne-50.png?" width="250">|<img src="ECG-time-series/visualization/mitbih-umap.png?" width="250">|<img src="ECG-time-series/visualization/mitbih-pca.png?" width="250">|
|**PTBDB**|<img src="ECG-time-series/visualization/ptbdb-tsne-50.png?" width="250">|<img src="ECG-time-series/visualization/ptbdb-umap.png?" width="250">|<img src="ECG-time-series/visualization/ptbdb-pca.png?" width="250">|

</center>

Requirements: pandas, numpy, scikit-learn, keras, matplotlib, xgboost, umap-learn

## Project 2 - TBA

## Project 3 - TBA

## Project 4 - TBA

### References

[1] Mohammad Kachuee, Shayan Fazeli, and Majid Sarrafzadeh. "ECG Heartbeat Classification: A Deep Transferable Representation." [arXiv preprint arXiv:1805.00794 (2018) ](https://arxiv.org/abs/1805.00794).

[2] CVxTz's GitHub implementation: ECG_Heartbeat_Classification ([link](https://github.com/CVxTz/ECG_Heartbeat_Classification))

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
