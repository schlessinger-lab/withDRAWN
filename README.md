 Drugs with unexpected toxicity post approval cause immense risk to patients before regulatory agencies detect and withdraw them. This repository contains all of the files needed to train an ensemble classifier to predict the withdrawal status of a drug and apply this classifier to drugs currently in clinical trials. We provide data sets containing features of drugs, drug fingerprints, drug targets, and drug target features. Additionally we proved the code for the classifiers trained individually on these subsets as well as the ensemble.

Classifiers hyperparameter tuned with a genetic algorithm on each individual set of features (protein targets, protein structure features, chemical fingerprints, chemical features) yielded the following accuracies on 10-fold cross validated holdout test sets respectively: 62.8%, 84.3%, 68.3%, and 77.0%. An ensemble approach, using a hyperparameter tuned k nearest neighbor classifier trained on the predictions from the above classifiers, achieved a 92% accuracy and 84.5% Matthews Correlation Coefficient. This ensemble predictor applied to compounds currently in clinical trials reveals candidates that are likely to be withdrawn.

### Data Files

- data_headers.xlsx: contains the headers for all the data files
- drug_features.csv: contains the chemical properties of the drugs
- drug_info.csv: contains the ATC assigned to drugs
- fp.csv: contains drug bitwise morgan fingerprint information
- sages.csv: contains drug target feature information
- targetsall.csv: contains IC50 values of protins that a drug inhibits
- tox_labels.csv: contains the labels of drug withdrawn status used in training and testing


### Code Dependancies

- python 3
- numpy
- pandas
- sklearn
- tpot