# Project Overview

This project analyzes unlabeled financial anomaly data to predict potentially fraudulent transactions. The project is an unsupervised machine learning task, utilizing Gaussian Mixture Models and Isolation Forests for anomaly detection. Given the lack of labeled anomaly data, evaluation of model performance was restricted to visual inspection and clustering metrics, such as the Silhouette score, the Calinski-Harabasz Index, and the Davies-Bouldin Index. The limitations of the dataset and of these scoring metrics are discussed in the [report](https://github.com/JoshuaGottlieb/Financial-Anomaly-Data/blob/main/docs/Gottlieb_Joshua-CS667-Project-03-Report.pdf) and [project notebook](https://github.com/JoshuaGottlieb/Financial-Anomaly-Data/blob/main/src/Gottlieb_Joshua-CS667-Project-03-Anomaly_Detection.ipynb). All work was performed in a single notebook, with the help of custom functions available under [src/modules](https://github.com/JoshuaGottlieb/Financial-Anomaly-Data/tree/main/src/modules). The project process consisted of four main steps: feature extraction and engineering, labeling of supsected anomalies, model training, and model evaluation.

Basic temporal features were extracted from the provided Timestamp column, such as month, day, hour, minute, and a binary weekend flag. Advanced temporal features were created using the Timestamp, Account, and Merchant columns to capture suspicious aggregated account activity which may be invisible from individual transactions. These advanced temporal features included time and amount deltas between transactions and hourly transaction counts and sums, as grouped by account or merchant, to pick up periods of high transaction flow which may appear non-anomalous individually but are potentially fraudulent in aggregation. Collinear features were removed using the variance inflation factor.

Suspected anomalies were identified by projecting the data to two dimensions using PCA and through the use of percentile analysis. Potentially anomalous datapoints were found in the ammount, account hourly transaction sums, and merchant hourly transaction sums based on exceptionally high values for data beyond the 99.66th percentile. These anomalies were confirmed through visual analysis. While visual analysis is not the most robust method for identifying patterns, in the absence of labeled data, it is one of the few tools available for analysis. Fortunately, the visual distribution of these anomalous data points matched intuition.

Each model was trained using 40 hyperparameter combinations. As there is no singular metric to use for scoring each combination, all models were saved for future evaluation using clustering metrics and visual evaluation. After training, the most promising models by clustering metrics were selected for visual analysis using PCA. The best performing models were the Gaussian Mixture Models using shared covariance matrices. These models were able to capture the majority of the suspected anomalies and were sometimes even able to capture points that seemed anomalous by visual inspection but were unable to be captured during exploratory data analaysis. The Isolation Forest models struggled to capture the suspected anoamlies and performed poorly on the dataset. This is due to the low variance present in the categorical variables, as these variables did not cleanly isolate data points during splitting. Since Isolation Forests choose the feature to split on randomly, these low quality variables led to poor anomaly isolation.

# Repository Structure

## Repository Structure
```
.
|── data/                                                              # Raw and processed data
|── docs/                                                              # Project report
|── models/                                                            # Pickled and compressed trained scikit-learn models
|── results/                                                           # Model prediction and clustering metric CSV files
|
|── src/                                                                    
|   ├── Gottlieb_Joshua-CS667-Project-03-Anomaly_Detection.ipynb       # Project notebook containing all preprocessing, EDA, model training, and model evaluation
|   └── modules/                                                            
|   	├── io_utils.py                                                # Functions for loading and saving data
|   	├── plotting.py                                                # Functions for creating informative plots
|   	├── plotting_utils.py                                          # Utility functions for formatting plots
|   	├── preprocessing.py                                           # Functions for extracting and encoding features
|   	├── statistics.py                                              # Functions for calculating statistical measures (VIF, Kruskal-Wallis)
|   	└── training.py                                                # Function for training models
|
|── visualizations/                                                    # Images saved from notebook for use in project report
```


# Libraries and Requirements

This project was performed using Python 3.12.3. The library versions used in this project can be installed with the following command:
```
pip install -r requirements.txt
```

The requirements are also listed below:
```
matplotlib==3.10.6
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
seaborn==0.13.2
```
