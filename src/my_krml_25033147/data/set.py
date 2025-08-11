from pandas import read_csv, DataFrame
from typing import Optional
from os.path import join
from os import listdir

def load_dataset(
    index_column : Optional[str],
    folder_path : str = './data/processed'
) -> dict:
    """
    loads the different dataset splits on a folder
    arguments
        path (str) folder path that stores the different csv files
    returns
        dict [file_name : DataFrame] : list of features and target variable stored on the folder
    """
    try: 
        if len(listdir(path = folder_path)) == 0:
            raise Exception("the folder is empty")
        files = [join(folder_path, f) for f in listdir(folder_path) if f.endswith('.csv')]
        if len(files) == 0:
            raise Exception(f"there are no '.csv' files on {folder_path}")
        data = {
            fp.split(sep = '/')[-1].split(sep = '.')[0] : read_csv(fp, index_col = index_column) for fp in files
        }
    except Exception as e:
        raise e
    return data
    