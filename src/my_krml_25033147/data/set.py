from pandas import read_csv, DataFrame, Series, to_datetime, date_range, Timedelta
from os.path import join
from os import listdir
from sklearn.model_selection import train_test_split
from os.path import exists
from openmeteo_requests import Client
from typing import List, Tuple

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

def time_data_split(
    df: DataFrame,
    target_col: str,
    test_ratio: float = 0.15
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits the DataFrame into training, validation, and test sets.

    Parameters:
    - df (DataFrame): The input DataFrame containing features and the target column.
    - target_col (str): The name of the target column to be separated from the features.
    - test_ratio (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.15.

    Returns:
    - Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]: 
      The training, validation, and test sets for features and target, in the following order:
      (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    if not 0 < test_ratio < 0.5:
        raise ValueError("test_ratio must be between 0 and 0.5")

    df_copy = df.copy()
    if target_col not in df_copy.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")

    target = df_copy.pop(item=target_col)
    test_size = int(df_copy.shape[0] * test_ratio)
    train_size = df_copy.shape[0] - (2 * test_size)

    # train split
    X_train = df_copy.iloc[:train_size, :]
    y_train = target.iloc[:train_size]

    # validation split
    X_val = df_copy.iloc[train_size:train_size + test_size, :]
    y_val = target.iloc[train_size:train_size + test_size]

    # test split
    X_test = df_copy.iloc[train_size + test_size:, :]
    y_test = target.iloc[train_size + test_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test
 
class WeatherDataFetcher:
    def __init__(
        self,
        start_date : str,
        end_date : str,
        latitude : float = -33.8678,
        longitude : float = 151.2073,
        variables_list : List[str] = ["weather_code", "temperature_2m_mean", "apparent_temperature_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "sunshine_duration", "daylight_duration", "precipitation_sum", "rain_sum", "precipitation_hours", "cloud_cover_mean", "dew_point_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean", "surface_pressure_mean", "wind_gusts_10m_mean", "wind_speed_10m_mean"],
        time_zone : str = "Australia/Sydney"
    ) -> None:
        # attributes of the class
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date
        self.end_date = end_date
        self.variables_list = variables_list
        self.time_zone = time_zone
    # methods
    def get_data(self) -> DataFrame:
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
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "daily": self.variables_list,
                "timezone": self.time_zone
            }
            responses = open_meteo.weather_api(url, params = params)
            response = responses[0]
            daily = response.Daily()
            daily_data = {
                value : daily.Variables(index).ValuesAsNumpy() for index, value in enumerate(self.variables_list)
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
            # df.dropna(inplace = True) # This is the way to get the full year
            df.sort_values(by = 'date', ascending = True, inplace = True)
            df.set_index(keys = 'date', inplace = True)
            return df
        except Exception as e:
            raise Exception(f"Error extracting historical data: {e}")
        
    def generate_lagged_features(
        self,
        df: DataFrame,
        features_list: List[str],
        lagging_period: int = 7
    ) -> DataFrame:
        """
        Generates lagged features for a given DataFrame.

        This function creates new columns in the DataFrame for each feature in the `features_list`, 
        representing the values of these features from previous time steps. The number of lagged 
        time steps is determined by the `lagging_period`.

        Parameters:
        - df (DataFrame): The input DataFrame containing the original features.
        - features_list (List[str]): A list of column names in the DataFrame for which lagged features 
          should be generated.
        - lagging_period (int, optional): The number of previous time steps to include as lagged features. 
          Defaults to 7.

        Returns:
        - DataFrame: A new DataFrame with the original features and additional lagged features.
        """
        df_copy = df.copy()
        shifted_columns = {
            f"{col}_t-{i}": df[col].shift(i) for i in range(1, lagging_period + 1)
            for col in features_list
        }
        df_shifted = df_copy.assign(**shifted_columns)
        # removing duplicate values after shifting the features
        df_shifted.dropna(inplace = True)    
        return df_shifted

class WeatherDataClassification(WeatherDataFetcher):
    # methods
    def create_rain_column(
        self,
        df : DataFrame,
        rain_wmo_codes : List[int] = [61, 62, 63],
    ) -> DataFrame:
        """
        creates a boolean column indicating whether it rained based on WMO weather codes
        
        args:
            df (DataFrame): dataframe containing weather data with 'weather_code' column
            rain_wmo_codes (List[int]): list of WMO weather codes that indicate rain
                                    (default: [61, 62, 63])
        
        returns:
            df (DataFrame): dataframe with the rain column
            
        raises:
            KeyError: if 'weather_code' column is missing from the dataframe
            Exception: if there's an error processing the weather codes
            
        note:
            - WMO codes 61-64: rain intensity (light to heavy)
            - WMO codes 80-82: rain showers (light to heavy)
            - Returns a boolean mask that can be used for filtering or analysis
        """
        try:
            df_copy = df.copy()
            if 'weather_code' not in df.columns:
                raise KeyError(f"the df does not have 'weather_code' column")
            df_copy['rain'] = df['weather_code'].isin(rain_wmo_codes)
            return df_copy
        except Exception as e:
            raise Exception(f"Error while creating the 'rain' column: {e}")
        
    def generate_rain_in_future(
        self,
        df : DataFrame,
        moving_window : int = 7
    ) -> DataFrame:
        """
        Applies a rolling window to the input series and shifts the result to create a target look-ahead column.

        This function is used to generate a new column, `will_rain_in_7_days`, which indicates whether there is any 
        occurrence of a specified condition within a 7-day moving window. The result is shifted to align with the 
        start of the window.

        Parameters:
        - df (DataFrame): The input pandas DataFrame to apply the rolling window on.
        - moving_window (int, optional): The size of the moving window. Defaults to 7.

        Returns:
        - DataFrame: A new pandas DataFrame with the look-ahead target values.
        """
        df_copy = df.copy()
        df_copy[f"will_rain_in_{moving_window}_days"] = df_copy['rain'].rolling(window = moving_window).apply(lambda x: any(x)).shift(periods = - moving_window)
        return df_copy