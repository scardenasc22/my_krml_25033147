from pandas import read_csv, DataFrame
from typing import Optional
from os.path import join
from os import listdir

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
    