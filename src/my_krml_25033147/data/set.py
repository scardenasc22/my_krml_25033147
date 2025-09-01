from pandas import read_csv, DataFrame, Series, to_datetime, date_range, Timedelta
from os.path import join
from os import listdir
from sklearn.model_selection import train_test_split
from os.path import exists
from openmeteo_requests import Client
from typing import List

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

def get_open_meteo_daily_data(
    lat: float,
    long: float,
    start_date: str,
    end_date: str,
    variables_list: List[str],
    time_zone: str = "Australia/Sydney",
) -> DataFrame:
    """
    retrieves historical daily weather data from Open-Meteo API for a specific location and date range
    
    args:
        lat (float): latitude coordinate of the location
        long (float): longitude coordinate of the location
        start_date (str): start date in YYYY-MM-DD format
        end_date (str): end date in YYYY-MM-DD format
        variables_list (List[str]): list of weather variables to retrieve (e.g., ['temperature_2m_max', 'precipitation_sum'])
        time_zone (str): timezone for the data (default: "Australia/Sydney")
    
    returns:
        DataFrame: daily weather data with columns for each variable and a 'date' column
        
    raises:
        Exception: if there's an error retrieving data from the Open-Meteo API
        
    note:
        - Data is automatically cleaned by removing rows with missing values
        - Date column is converted to pandas datetime format
        - Uses Open-Meteo's historical weather archive API
        - Common variables include: 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max'
    """
    try:
        open_meteo = Client()
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": long,
            "start_date": start_date,
            "end_date": end_date,
            "daily": variables_list,
            "timezone": time_zone
        }
        responses = open_meteo.weather_api(url, params = params)
        response = responses[0]
        daily = response.Daily()
        daily_data = {
            value : daily.Variables(index).ValuesAsNumpy() for index, value in enumerate(variables_list)
        }
        daily_data['date'] = date_range(
            start = to_datetime(daily.Time(), unit = "s", utc = True),
            end = to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )
        df = DataFrame(
            data = daily_data
        )
        df.dropna(inplace = True)
        return df
    except Exception as e:
        raise Exception(f"Error extracting historical data: {e}")

def create_rain_column(
    df : DataFrame,
    rain_wmo_codes : List[int] = None
) -> Series:
    """
    creates a boolean column indicating whether it rained based on WMO weather codes
    
    args:
        df (DataFrame): dataframe containing weather data with 'weather_code' column
        rain_wmo_codes (List[int]): list of WMO weather codes that indicate rain
                                   (default: [61, 62, 63, 64, 80, 81, 82])
    
    returns:
        Series: boolean series where True indicates rain occurred
        
    raises:
        KeyError: if 'weather_code' column is missing from the dataframe
        Exception: if there's an error processing the weather codes
        
    note:
        - WMO codes 61-64: rain intensity (light to heavy)
        - WMO codes 80-82: rain showers (light to heavy)
        - Returns a boolean mask that can be used for filtering or analysis
    """
    if rain_wmo_codes is None:
        rain_wmo_codes = list(range(61, 66)) + list(range(80, 83))
    try:
        if 'weather_code' not in df.columns:
            raise KeyError(f"the df does not have 'weather_code' column")
        rain_col = df['weather_code'].isin(rain_wmo_codes)
        return rain_col
    except Exception as e:
        raise Exception(f"Error while creating the 'rain' column: {e}")