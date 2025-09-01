from pandas import read_csv, DataFrame, Series
from os.path import join
from os import listdir
from sklearn.model_selection import train_test_split
from os.path import exists

def load_dataset(
    folder_path: str = './data/processed',
    **read_csv_kwargs
) -> dict:
    """
    loads the different dataset splits on a folder
    
    args:
        folder_path (str): folder path that stores the different csv files
        **read_csv_kwargs: additional arguments passed to pandas.read_csv()
    
    returns:
        dict: {file_name: DataFrame} mapping of features and target variables
    """
    try:
        if len(listdir(path=folder_path)) == 0:
            raise Exception("the folder is empty")
            
        files = [join(folder_path, f) for f in listdir(folder_path) if f.endswith('.csv')]
        if len(files) == 0:
            raise Exception(f"there are no '.csv' files on {folder_path}")
            
        data = {
            fp.split(sep='/')[-1].split(sep='.')[0]: read_csv(fp, **read_csv_kwargs) 
            for fp in files
        }
        
    except Exception as e:
        raise Exception(f"Error loading datasets: {e}") from e
        
    return data

def train_test_val_split(
    features: DataFrame,
    target: Series,
    test_ratio: float = 0.2
) -> tuple[DataFrame, Series, DataFrame, Series, DataFrame, Series]:
    """
    splits features and target into training, validation, and test sets
    
    args:
        features (DataFrame): feature variables to split
        target (Series): target variable to split
        test_ratio (float): proportion of data to use for test set (default: 0.2)
    
    returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test) where:
            - X_train (DataFrame): training features
            - y_train (Series): training targets
            - X_val (DataFrame): validation features  
            - y_val (Series): validation targets
            - X_test (DataFrame): test features
            - y_test (Series): test targets
    note:
        The function first splits data into test and remaining data using test_ratio.
        Then splits the remaining data into train and validation sets.
        Uses random_state=1 for reproducible splits.
    """
    val_ratio = test_ratio / (1 - test_ratio)
    X_data, X_test, y_data, y_test = train_test_split(
        features,
        target,
        test_size = test_ratio,
        random_state = 1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_data,
        y_data,
        test_size = val_ratio,
        random_state = 1
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_datasets(
    X_train: DataFrame, 
    y_train: Series, 
    X_val: DataFrame, 
    y_val: Series, 
    X_test: DataFrame, 
    y_test: Series,
    target_path: str
) -> None:
    """
    saves all dataset splits (train, validation, test) to CSV files in the specified directory
    
    args:
        X_train (DataFrame): training features
        y_train (Series): training targets
        X_val (DataFrame): validation features
        y_val (Series): validation targets
        X_test (DataFrame): test features
        y_test (Series): test targets
        target_path (str): directory path where to save the CSV files
    
    returns:
        None: saves files to disk
        
    raises:
        Exception: if the target directory doesn't exist
        
    note:
        Files will be saved as:
        - X_train.csv, y_train.csv
        - X_val.csv, y_val.csv  
        - X_test.csv, y_test.csv
    """
    try:
        if not exists(target_path):
            raise Exception(f"the folder '{target_path}' does not exist")
        
        # Save training datasets
        X_train.to_csv(path_or_buf=join(target_path, 'X_train.csv'), index=False)
        y_train.to_csv(path_or_buf=join(target_path, 'y_train.csv'), index=False)
        
        # Save validation datasets
        X_val.to_csv(path_or_buf=join(target_path, 'X_val.csv'), index=False)
        y_val.to_csv(path_or_buf=join(target_path, 'y_val.csv'), index=False)
        
        # Save test datasets
        X_test.to_csv(path_or_buf=join(target_path, 'X_test.csv'), index=False)
        y_test.to_csv(path_or_buf=join(target_path, 'y_test.csv'), index=False)
        
    except Exception as e:
        raise Exception(f"Error saving datasets: {e}") from e