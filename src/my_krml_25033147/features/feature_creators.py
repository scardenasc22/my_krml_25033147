from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List

class LaggedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer to create lagged features for a DataFrame.
    """
    def __init__(self, features_list, lagging_period):
        self.features_list = features_list
        self.lagging_period = lagging_period

    def fit(self, X, y=None):
        """
        This transformer does not need to be fitted, so we simply return self.
        """
        return self

    def transform(self, X):
        """
        Creates lagged features for the specified columns in the input DataFrame.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            
        Returns:
            pd.DataFrame: A new DataFrame with the lagged features.
        """
        copy_df = X.copy()
        
        # Build a single dictionary of all lagged columns
        shifted_columns = {
            f"{col}_t-{i}": copy_df[col].shift(i) for i in range(1, self.lagging_period + 1)
            for col in self.features_list
        }

        # Use .assign() to add all the new columns at once
        df_with_lags = copy_df.assign(**shifted_columns)
        
        # The first `lagging_period` rows will have NaN values
        # after creating the lagged features, so we replace them using 0
        
        # Identify columns with missing values
        columns_with_nan = df_with_lags.columns[df_with_lags.isna().any()].tolist()

        # Fill NaN values in those columns with 0
        df_with_lags[columns_with_nan] = df_with_lags[columns_with_nan].fillna(0)
        
        return df_with_lags

class TimeFeatureCreator(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer to create time-based features 
    from a DataFrame's datetime index.
    """
    def fit(self, X, y=None):
        """
        This transformer does not need to be fitted, so we simply return self.
        """
        return self

    def transform(self, X):
        """
        Creates time-based features from the DataFrame's index.
        
        Args:
            X (pd.DataFrame): The input DataFrame with a datetime index.
            
        Returns:
            pd.DataFrame: A new DataFrame with the added time features.
        """
        # Ensure the index is a datetime index
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("The DataFrame index must be a DatetimeIndex.")
            
        copy_df = X.copy()
        
        copy_df['month'] = copy_df.index.month
        copy_df['year'] = copy_df.index.year
        copy_df['day_of_week'] = copy_df.index.weekday
        copy_df['day_of_month'] = copy_df.index.day
        copy_df['day_of_year'] = copy_df.index.day_of_year
        
        return copy_df

class RainFeatureCreator(BaseEstimator, TransformerMixin):
    """
    A custom transformer for creating a binary 'rain' feature based on weather codes.

    This transformer checks for the presence of a 'rain' column in the input DataFrame. If the 'rain' 
    column is not present, it creates a new 'rain' column based on specified weather codes that indicate 
    rain. The transformer is compatible with scikit-learn's pipeline and can be used in preprocessing steps.

    Attributes:
    - rain_codes (List[int]): A list of weather codes that indicate rain. Defaults to [61, 62, 63, 66, 67, 80, 81, 82].

    Methods:
    - fit(X, y=None): Returns the transformer instance. No fitting is required.
    - transform(X, rain_codes=List[int]): Transforms the input DataFrame by adding a 'rain' column if it doesn't exist.

    Raises:
    - KeyError: If the 'weather_code' column is not present in the input DataFrame.
    """

    def fit(
        self,
        X,
        y=None
    ):
        return self

    def transform(
        self,
        X: pd.DataFrame,
        rain_codes: List[int] = [61, 62, 63, 66, 67, 80, 81, 82]
    ):
        df_copy = X.copy()
        if "rain" in df_copy.columns:
            return df_copy
        elif not "weather_code" in df_copy.columns:
            raise KeyError(f"the 'weather_code' is not available")
        else:
            df_copy['rain'] = df_copy['weather_code'].isin(rain_codes).astype(int)
            return df_copy
        