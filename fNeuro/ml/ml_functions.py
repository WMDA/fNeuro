from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def estimate_model(model, X: np.array, y: np.array) -> dict:
    '''
    Function to estimate classification models.
    Cross validates to get AUC and accuracy scores

    Parameters
    ----------
    model: sckitlearn model
        instance of model to get cross-val scores
        from
    X: np.array
        Features for the classification model
    y: np.array
       Predictor for classification model

    Returns
    -------
    dict: dictionary 
        dict of accuarcy and accuracy vals

    '''
    cv = StratifiedShuffleSplit(n_splits=10, random_state=0, test_size=0.2, train_size=0.8)
    return {
        'Accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=2),
        'ROC_AUC': cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=2),
    }

def estimate_regression_model(model: object, X: np.array, y: np.array) -> dict:
    
    '''
    Function to estimate regression models with cross validation

    Parameters
    ----------
    model: object 
        instance of sckit learn model
    
    X: np.array
        array of X values

    y: np.array of y values
        array of y values

    Returns
    -------
    dict: dictionary object
        dict of r2, MAE and MSE values
    '''

    cv = StratifiedShuffleSplit(n_splits=10, random_state=42, test_size=0.2, train_size=0.8)
    try: 
        return {
        'r2': cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=3),
        'MAE': cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=3),
        'MSE': cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=3),
    }
    except Exception as e:
        print(e)