# Optimizing an ML Pipeline in Azure

## Overview
In this project, I build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
Used dataset is generally available [Source](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and it is data gathered from direct marketing campaigns by a bank.
The goal is to predict whether the customer will subscribe for a term deposit or not, based on age, job, education etc.

## Approach
### 1. Scikit-learn Pipeline
Steps taken:
1. Gather the data and convert it to dataset
2. Clean the data - using clean_data function
3. Train the chosen model - Logistic Regression from scikit-learn was chosen as a model for this task, training has taken place in AzureML Studio workspace
4. Choose the Hyperparameters - Next step was to use Hyperdrive in AzureML to choose the best performing hyperparameters:
    * Definition of parameters space - for Inverse of regularization strength(C) and Maximum number of iterations (max_iter). **Inverse of regularization strength:** Default 1, lower might cause underfitting, higher overfitting of the model. Choice was used between 0.01 and 10000. **Maximum iterations:** Default 100
    * Parameter sampling - There are three possible algorithms: **Bayesian Sampling**, **Grid Sampling** and **Random Sampling**, and the last one was used. **Bayesian Sampling:** samples are based on previous samples' performance, does not support early termination policies. **Grid Sampling:** Grid search over all possibilities, supports early termination for lower performance runs. **Random Sampling:** Randomly selects values for hyperparameters. 
    * Early stopping policy - improves utilization and efficiency by termination of worse runs. 

Hyperparameter runs:
![img](/img/1.PNG)
![img](/img/2.PNG)

### 2. AutoML
Steps taken:
1. Gather the data and convert it to dataset
2. Clean the data - it required some changes in the code to get and combine test/train data and convert it to TabularDataset
3. Choose the model:
    * task: classification
    * metric: accuracy
    * training_data: training dataset
    * label_column: y
4. Train models:
    ![img](/img/aml1.PNG)
    AutoML has chosen Ensemble models as the best ones and XGBoost as a single model.
    ![img](/img/aml2.PNG)
    AML has also indicated that there are unbalanced classes in the provided dataset.
    ![img](/img/aml3.PNG)
    ![img](/img/aml4.PNG)
    Top features chosen by Azure AutoML were duration, nr.employed, emp.var.rate and euribor3m.
    ![img](/img/aml5.PNG)


## Pipeline comparison
**Hyperparameter Tuning**
![img](/img/hyperparameter.png)
**AutoML**
![img](/img/automl.png)

The main difference between these two approaches was that AutoML has checked every model possible for the given task, whereas Hyperparameter tuning was just searching through different variations of the single model. In theory I think Hyperparameter tuning should be easier and faster, but it relies heavily on chosen parameters, policies etc.

![img](/img/hyperparameters.PNG)
![img](/img/xgb.PNG)
![img](/img/votingensemble.PNG)

Only VotingEnsemble model performed better than hyperparemeter pipeline. We could re-configure AutoML to use the same validation set for calculating accuracy and probably we would receive more than one model with accuracy > .9135712

The best run for hyperparameter pipeline set --C to 19.417 and --max_iter to 1000.


## Future work
AutoML could be tuned using in-built features like featurization or n-cross validation just to make sure there was no data skew in train/test dataset.
Additional data could also improve the performance of both pipelines.
