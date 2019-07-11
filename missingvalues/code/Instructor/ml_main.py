"""
Author      : Yi-Chieh Wu
Class       : HMC CS 181R
Date        : 2019 June 7
Description : ML Pipeline

With modifications by Huey Fields
"""

# python modules
import os
import argparse

# numpy, pandas, and sklearn modules
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# custom classifier and dataset libraries
import classifiers, preprocessors, datasets

RESULTS_FOLDER = os.path.join("..", "..", "results")
N_ITER = 100    # number of parameter settings sampled (trade-off runtime vs quality)
CV = 10        # number of folds in cross-validation
FEATURES_TO_SHOW = 10   # number of features to print for each binary task

def report_metrics(y_true, y_pred, labels=None, target_names=None, write_file = None):
    """Report main classification metrics.

    Parameters
    --------------------
        y_true       -- ground truth (correct) target values, array of shape (n_samples,)
        y_pred       -- estimated target values returned by classifier, array of shape (n_samples,)
        labels       -- list of label indices to include in report, list of strings
        target_names -- display names matching labels (same order), list of strings
        file         -- optional file to write to

    Return
    --------------------
        C      -- confusion matrix, see sklearn.metrics.confusion_matrix
        a      -- accuracy score, see sklearn.metrics.accuracy_score
        p      -- precision score, see sklearn.metrics.precision_score
        r      -- recall score, see sklearn.metrics.recall_score
        f1     -- f1 score, see sklearn.metrics.f1_score
    """

    # confusion matrix, then wrap in pandas to pretty print
    C = metrics.confusion_matrix(y_true, y_pred, labels)
    df = pd.DataFrame(C, columns=target_names, index=target_names)
    print("Confusion Matrix\n", df)
    print()

    # accuracy
    a = metrics.accuracy_score(y_true, y_pred)
    print("accuracy: ", a)
    print()

    # area under receiver operating characteristic curve
    auroc = metrics.roc_auc_score(y_true, y_pred)
    print("auroc: ", auroc)
    print()

    # precision, recall, f1
    p, r, f1, s = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                          average="weighted")
    # print report (redundant with above but easier)
    report = metrics.classification_report(y_true, y_pred, labels, target_names)
    print(report)

    if write_file:
        write_file.write("Confusion Matrix\n" + str(df) + "\n")
        write_file.write("accuracy: " + str(a) + "\n")
        write_file.write("auroc: " + str(auroc) + "\n")
        write_file.write(str(report) + "\n")

    return C, (a, p, r, f1)

def make_pipeline(preprocessor_list, classifier, n, d):
    """Make ML pipeline.

    Parameters
    --------------------
        preprocessor_list -- preprocessors, list of strings
        classifier        -- classifier, string
        n                 -- number of samples, int
        d                 -- number of features, int
    """
    steps = []
    param_grid = {}

    # get preprocessor(s) and hyperparameters to tune using cross-validation
    for preprocessor in preprocessor_list:
        pp = getattr(preprocessors, preprocessor)()
        name = type(pp).__name__
        transform = pp.transformer_
        steps.append((name, transform))
        for key, val in pp.param_grid_.items():
            param_grid[name + "__" + key] = val

    # get classifier and hyperparameters to tune using cross-validation
    clf = getattr(classifiers, classifier)(n,d)
    name = type(clf).__name__
    transform = clf.estimator_
    steps.append((name, transform))
    for key, val in clf.param_grid_.items():
        param_grid[name + "__" + key] = val

    # stitch together preprocessors and classifier
    pipe = Pipeline(steps)
    return pipe, param_grid

def runML(dataset = "sqf", preprocessor_list = [], classifier = "LogReg", multiclass = False, scoring = None, param_tuning = True,\
    file_suffix = "", **kwargs):
    """Run ML pipeline.

    Parameters
    --------------------
        dataset           -- dataset, string
        preprocessor_list -- preprocessors, list of strings
        classifier        -- classifier, string
        multiclass        -- whether or not to predict multiclass option of dataset, boolean
        scoring           -- the scoring metric to use for CV, string
        param_tuning      -- whether or not to tune hyperparameters before predictions are made, boolean
        **kwargs          -- hyperparameter values to be passed to model if param_tuning is set to False
    """

    # get dataset, then split into train and test set
    X, y, labels, target_names, feature_names = getattr(datasets, dataset)(multiclass = multiclass)

    print("Dataset Loaded")

    # Set train/test array
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    n,d = X_train.shape

    #print(X_train.isnull().sum())

    print("Number of features:", d)

    # make pipeline
    pipe, param_grid = make_pipeline(preprocessor_list, classifier, n, d)

    # If option is set, conduct hyperparameter tuning
    if param_tuning:
        # get param grid size
        sz = 1
        for vals in param_grid.values():
            sz *= len(vals)

        # tune model using randomized search
        n_iter = min(N_ITER, sz)    # cap max number of iterations

        # Determine scoring method
        if scoring:
            scoring = metrics.make_scorer(getattr(metrics, scoring))

        search = RandomizedSearchCV(pipe, param_grid, n_iter=n_iter, scoring = scoring, cv=CV)
        print("Beginning hyperparameter tuning via cross-validation")
        search.fit(X_train, y_train)

        # set model and params to be used
        pipe = search.best_estimator_
        params = search.best_params_

        print("Best parameters set found on development set:\n")
        print(params)
        print("\n")

    # Otherwise read keyword arguments to find hyperparams
    else:
        clf = pipe.named_steps[classifier]
        params = {}
        
        # Set each hyperparameter and save them in a dictionary to be written to file
        for param, value in kwargs.items():
            setattr(clf, param, value)
            params[classifier + "__" + param] = value

        # Fit model
        pipe.fit(X_train, y_train)

    clf = pipe.named_steps[classifier]

    # Create filename including dataset, number of features, hyperparameters, and whether single or multiclass
    filename = dataset + "_" + str(d) + "_features_" 
    for param, value in params.items():
        filename += param + "=" + str(value) + "_"
    
    if multiclass:
        filename += "multiclass"

    else:
        filename += "singleclass"

    # Append file_suffix, if specified
    if file_suffix != "":
        filename += "_" + file_suffix
    
    # Append file extension
    filename += ".txt"

    f = open(os.path.join(RESULTS_FOLDER, filename), "w")

    # Report feature importances
    features = []

    if classifier == "LogReg":
        # Report for each binary task
        for task in range(len(clf.coef_)):
            print(str(labels[task]) + ":")
            f.write(str(labels[task]) + ":\n")

            # Print info for the most FEATURES_TO_SHOW most important features
            # Record full information in results file
            sorted_indices = np.argsort(clf.coef_)[task]

            for i in range(len(sorted_indices)):
                index = sorted_indices[i]

                if i > d - FEATURES_TO_SHOW:
                    print(feature_names[index], clf.coef_[task][index])

                f.write(feature_names[index] + ": " + str(clf.coef_[task][index]) + "\n")
                features += [feature_names[index]]

            print()
            f.write("\n")

    elif classifier == "DT" or classifier == "RF":
        # Print info for the most FEATURES_TO_SHOW most important features
        # Record full information in results file
        sorted_indices = np.argsort(clf.feature_importances_)

        for i in range(len(sorted_indices)):
            index = sorted_indices[i]

            if i > d - FEATURES_TO_SHOW:
                print(feature_names[index], clf.feature_importances_[index])

            f.write(feature_names[index] + ": " + str(clf.feature_importances_[index]) + "\n")
            features += [feature_names[index]]

    # report results
    print("Detailed classification report (training set):\n")
    f.write("Detailed classification report (training set):\n")
    y_true, y_pred = y_train, pipe.predict(X_train)
    res_train = report_metrics(y_true, y_pred, labels, target_names, f)
    print("\n")

    print("Detailed classification report (test set):\n")
    f.write("\nDetailed classification report (test set):\n")
    y_true, y_pred = y_test, pipe.predict(X_test)
    res_test = report_metrics(y_true, y_pred, labels, target_names, f)

def main():
    runML(dataset = "credit", preprocessor_list = ["Scaler", "Imputer"], classifier = "DT", scoring = "roc_auc_score", \
        file_suffix = "nodrops")

if __name__ == "__main__":
    main()