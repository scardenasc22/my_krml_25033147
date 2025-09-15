from sklearn.metrics import (
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay, 
    PrecisionRecallDisplay,
    accuracy_score, 
    recall_score, 
    f1_score,
    average_precision_score,
    root_mean_squared_error,
    mean_absolute_error
)
from sklearn.base import BaseEstimator
from pandas import DataFrame, Series
from typing import Tuple, Optional, List
from matplotlib.pyplot import subplots, tight_layout, suptitle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from seaborn import despine

def rmse_comparison(
    model: BaseEstimator,
    train_features : DataFrame,
    val_features : DataFrame,
    train_target : Series,
    val_target : Series
) -> DataFrame:
    """
    Compares the root mean squared error (RMSE) and the Mean Absolute Error (MAE) between training and validation datasets.

    This function takes a fitted scikit-learn compatible regressor and computes the RMSE
    on both the training and validation data. It returns a DataFrame that includes the RMSE
    scores for each dataset and the difference between them, which is useful for identifying
    overfitting or underfitting.

    Parameters:
    - model (BaseEstimator): A fitted scikit-learn compatible regressor (e.g., LinearRegression, RandomForestRegressor).
    - train_features (DataFrame): A pandas DataFrame containing the features of the training set.
    - val_features (DataFrame): A pandas DataFrame containing the features of the validation set.
    - train_target (Series): A pandas Series containing the target values of the training set.
    - val_target (Series): A pandas Series containing the target values of the validation set.

    Returns:
    - DataFrame: A pandas DataFrame with RMSE and MAE as the index, and columns for 'train', 'validation', and 'diff'.
    """
    # predictions
    pred_train = model.predict(train_features)
    pred_val = model.predict(val_features)
    results = {}
    results['train'] = []
    results['validation'] = []
    # appending the rmse scores
    results['train'].append(round(root_mean_squared_error(y_true = train_target, y_pred = pred_train), 4))
    results['validation'].append(round(root_mean_squared_error(y_true = val_target, y_pred = pred_val), 4))
    # appending the mae scores
    results['train'].append(round(mean_absolute_error(y_true = train_target, y_pred = pred_train), 4))
    results['validation'].append(round(mean_absolute_error(y_true = val_target, y_pred = pred_val), 4))
    # transforming the results into a dataframe
    results = DataFrame(data=results, index=['rmse', 'mae'])
    results['diff'] = results['train'] - results['validation']
    return results

def regression_preds_comparison(
    model: BaseEstimator,
    train_features : DataFrame,
    val_features : DataFrame,
    train_target : Series,
    val_target : Series,
    figure_size : Tuple[int, int] = (10, 5),
    title : Optional[str] = "Predictions comparison"
) -> Tuple[Figure, Axes]:
    """
    Visualizes the comparison between actual and predicted values for training and validation datasets.

    This function generates a scatter plot comparing the actual and predicted values for both the 
    training and validation datasets. It includes a line representing perfect predictions for reference.

    Parameters:
    - model (BaseEstimator): A fitted scikit-learn compatible regressor (e.g., LinearRegression, RandomForestRegressor).
    - train_features (DataFrame): A pandas DataFrame containing the features of the training set.
    - val_features (DataFrame): A pandas DataFrame containing the features of the validation set.
    - train_target (Series): A pandas Series containing the target values of the training set.
    - val_target (Series): A pandas Series containing the target values of the validation set.
    - figure_size (Tuple[int, int], optional): The size of the figure as (width, height). Defaults to (10, 5).
    - title (str, optional): The title of the plot. Defaults to "Predictions comparison".

    Returns:
    - Tuple[Figure, Axes]: The matplotlib Figure and Axes objects for the plot.
    """
    fig, axis = subplots(nrows = 1, ncols = 2, figsize = figure_size)
    despine(fig)
    # predictions
    pred_train = model.predict(train_features)
    pred_val = model.predict(val_features)
    # making the plot
    axis.plot(train_target, train_target, linestyle ='--', color ='black', label ='perfect prediction')
    axis.scatter(train_target, pred_train, color ='#25ADC2', edgecolors = 'white', label = 'training prediction')
    axis.scatter(val_target, pred_val, color ='#C98CAC', edgecolors = 'white', label = 'validation prediction')
    axis.legend()
    axis.set(x_label = "Actual", y_label = "Prediction")
    suptitle(title)
    return fig, axis
    
def classification_metrics_comparison(
    model: BaseEstimator,
    train_features : DataFrame,
    val_features : DataFrame,
    train_target : Series,
    val_target : Series
) -> DataFrame:
    """Compares classification metrics between training and validation datasets.

    This function takes a fitted scikit-learn compatible classifier and computes
    several key performance metrics on both the training and validation data. It
    returns a DataFrame that includes the scores for each dataset and the
    difference between them, which is useful for identifying overfitting.

    Args:
        model: A fitted scikit-learn compatible classifier (e.g., LogisticRegression, RandomForestClassifier).
        train_features: A pandas DataFrame containing the features of the training set.
        val_features: A pandas DataFrame containing the features of the validation set.
        train_target: A pandas Series containing the target labels of the training set.
        val_target: A pandas Series containing the target labels of the validation set.

    Returns:
        A pandas DataFrame with metrics (accuracy, recall, f1_score, auc_roc, auc_pr)
        as index, and columns for 'train', 'validation', and 'diff'.
    """
    # probabilities
    pred_prob_train = model.predict_proba(train_features)
    pred_prob_val = model.predict_proba(val_features)
    # labels
    pred_train = model.predict(train_features)
    pred_val = model.predict(val_features)
    # results dictionary
    results = {}
    results['train'] = []
    results['validation'] = []
    # appending the values of accuracy
    results['train'].append(round(accuracy_score(y_true=train_target, y_pred=pred_train), 4))
    results['validation'].append(round(accuracy_score(y_true=val_target, y_pred=pred_val), 4))
    # appending the results of recall
    results['train'].append(round(recall_score(y_true=train_target, y_pred=pred_train), 4))
    results['validation'].append(round(recall_score(y_true=val_target, y_pred=pred_val), 4))
    # appending the results of f1 score
    results['train'].append(round(f1_score(y_true=train_target, y_pred=pred_train), 4))
    results['validation'].append(round(f1_score(y_true=val_target, y_pred=pred_val), 4))
    # appending the results of auc roc score
    results['train'].append(round(roc_auc_score(train_target, pred_prob_train[:, 1]), 4))
    results['validation'].append(round(roc_auc_score(val_target, pred_prob_val[:, 1]), 4))
    # appending the results of auc pr score
    results['train'].append(round(average_precision_score(train_target, pred_prob_train[:, 1]), 4))
    results['validation'].append(round(average_precision_score(val_target, pred_prob_val[:, 1]), 4))
    # transforming the results into a dataframe
    results = DataFrame(data=results, index=['accuracy', 'recall', 'f1_score', 'auc_roc', 'auc_pr'])
    results['diff'] = results['train'] - results['validation']
    return results
    
def roc_comparison(
    model: BaseEstimator,
    train_features : DataFrame,
    val_features : DataFrame,
    train_target : Series,
    val_target : Series,
    figure_size : Tuple[int, int] = (10, 5),
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
    figure_size : Tuple[int, int] = (10, 5),
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

def pr_curve_comparison(
    model: BaseEstimator,
    train_features: DataFrame,
    val_features: DataFrame,
    train_target: Series,
    val_target: Series,
    title: Optional[str] = "Precision-Recall Curves",
    figure_size: Tuple[int, int] = (10, 5),
) -> Tuple[Figure, Axes]:
    """
    Creates the precision-recall curves for the training and validation sets.

    Args:
        model (BaseEstimator): A fitted scikit-learn estimator that implements predict_proba method.
        train_features (DataFrame): Training features.
        val_features (DataFrame): Validation features.
        train_target (Series): Training target variable.
        val_target (Series): Validation target variable.
        figure_size (Tuple[int, int]): Figure size as (width, height).
        title (str): Main title for the plot.

    Returns:
        Tuple[Figure, Axes]: Figure and axes objects to save the plots.
    """
    fig, axis = subplots(nrows=1, ncols=2, figsize=figure_size)
    
    # Training Precision-Recall curve
    PrecisionRecallDisplay.from_estimator(
        estimator=model,
        X=train_features,
        y=train_target,
        ax=axis[0],
        name="Training"
    )
    axis[0].set(title="Precision-Recall Curve for Training")
    
    # Validation Precision-Recall curve
    PrecisionRecallDisplay.from_estimator(
        estimator=model,
        X=val_features,
        y=val_target,
        ax=axis[1],
        name="Validation"
    )
    axis[1].set(title="Precision-Recall Curve for Validation")
    
    suptitle(title)
    tight_layout()
    return fig, axis

def generate_probability_output(
    model : BaseEstimator,
    features : DataFrame,
    col_names : List[str] = ['non_drafted', 'drafted']
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