from sklearn.metrics import roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator
from pandas import DataFrame, Series
from typing import Tuple, Optional, List
from matplotlib.pyplot import subplots, tight_layout, suptitle
from matplotlib.figure import Figure
from matplotlib.axes import Axes

def auc_performance_comparison(
    model: BaseEstimator,
    train_features : DataFrame,
    val_features : DataFrame,
    train_target : Series,
    val_target : Series
) -> None:
    """
    computes the auc roc score for training, validation set, and the difference between the two
    args:
        model (BaseEstimator): A fitted scikit-learn estimator that implements predict_proba method
        train_features (DataFrame): Training features
        val_features (DataFrame): Validation features  
        train_target (Series): Training target variable
        val_target (Series): Validation target variable
    returns
        None : prints the comparison between the training and validation performance
    """
    pred_prob_train = model.predict_proba(train_features)
    pred_prob_val = model.predict_proba(val_features)
    train_auc = roc_auc_score(train_target, pred_prob_train[:, 1])
    val_auc = roc_auc_score(val_target, pred_prob_val[:, 1])
    print(f"training AUC: {train_auc:5f}\nvalidation AUC: {val_auc:5f}\ndifference: {(train_auc - val_auc):5f}")

def roc_comparison(
    model: BaseEstimator,
    train_features : DataFrame,
    val_features : DataFrame,
    train_target : Series,
    val_target : Series,
    figure_size : Tuple[int, int] = (16, 6),
    title : Optional[str] = "ROC curves"
) -> Tuple[Figure, Axes]:
    """
    creates the roc curve plot for the training and validation set
    args:
        model (BaseEstimator): A fitted scikit-learn estimator that implements predict_proba method
        train_features (DataFrame): Training features
        val_features (DataFrame): Validation features  
        train_target (Series): Training target variable
        val_target (Series): Validation target variable
        figure_size (Tuple[int, int]): Figure size as (width, height)
        title (str): Main title for the plot
    returns
        Tuple[Figure, Axes]: Figure and axes objects to save the plots
    """
    fig, axis = subplots(nrows = 1, ncols = 2, figsize = figure_size)
    
    # Training ROC curve
    RocCurveDisplay.from_estimator(
        estimator = model,
        X = train_features,
        y = train_target,
        ax = axis[0],
        name = "Training",
        color = "#9A607F"
    )
    axis[0].set(title = "ROC curve of training")
    
    # Validation ROC curve
    RocCurveDisplay.from_estimator(
        estimator = model,
        X = val_features,
        y = val_target,
        ax = axis[1],
        name = "Validation",
        color = "#006BA2"
    )
    axis[1].set(title = "ROC curve of validation")
    
    suptitle(title)
    tight_layout()
    return fig, axis

def conf_mat_comparison(
    model: BaseEstimator,
    train_features : DataFrame,
    val_features : DataFrame,
    train_target : Series,
    val_target : Series,
    title : Optional[str] = "Confusion Matrices",
    figure_size : Tuple[int, int] = (16, 6),
) -> Tuple[Figure, Axes]:
    """
    creates the confusion matrix for the training and validation set
    args:
        model (BaseEstimator): A fitted scikit-learn estimator that implements predict_proba method
        train_features (DataFrame): Training features
        val_features (DataFrame): Validation features  
        train_target (Series): Training target variable
        val_target (Series): Validation target variable
        figure_size (Tuple[int, int]): Figure size as (width, height)
        title (str): Main title for the plot
    returns
        Tuple[Figure, Axes]: Figure and axes objects to save the plots
    """
    fig, axis = subplots(nrows = 1, ncols = 2, figsize = figure_size)
    
    # Training ROC curve
    ConfusionMatrixDisplay.from_estimator(
        estimator = model,
        X = train_features,
        y = train_target,
        ax = axis[0],
        cmap = "Greens"
    )
    axis[0].set(title = "Confusion Matrix for training")
    
    # Validation ROC curve
    ConfusionMatrixDisplay.from_estimator(
        estimator = model,
        X = val_features,
        y = val_target,
        ax = axis[1],
        cmap = 'Reds'
    )
    axis[1].set(title = "Confusion Matrix for validation")
    
    suptitle(title)
    tight_layout()
    return fig, axis

def generate_probability_output(
    model : BaseEstimator,
    features : DataFrame,
    col_names : List[str, str] = ['non_drafted', 'drafted']
) -> DataFrame:
    """
    generates the prediction probabilities of a model based on a set of features 
    with the format required for the kaggle competition
    args
        model (BaseEstimator): A fitted scikit-learn estimator that implements predict_proba method
        features (DataFrame): features to make the predictions (can be either test, train or validation)
    returns
        DataFrame : a dataframe with the format required for the submission
    """
    prob_array = model.predict_proba(features)
    prob_df = DataFrame(
        data = prob_array,
        columns = col_names,
        index = features.index
    )
    prob_df = prob_df.drop(columns = prob_df.columns[0], axis = 1)
    return prob_df