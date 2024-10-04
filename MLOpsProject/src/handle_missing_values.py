import logging
from abc import ABC, abstractmethod
import pandas as pd

logging.basicConfig(level=logging.info, format = "%(asctime)s - %(levelname)s - %(message)s")

#base abstract class
class MissingValueHandlingStrategy(ABC):
    
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        
        pass
    

#concrete strategy 
class DropMissingValueStrategy(MissingValueHandlingStrategy):
    
    def __init__(self, axis = 0, thresh = None):
        
        self.axis = axis
        self.thresh = thresh
        
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"dropping missing values with axis= {self.axis} and thresh = {self.thresh}")
        df_cleaned = df.dropna(axis = self.axis, thresh = self.thresh)
        logging.info("missing values dropped from df")
        return df_cleaned
    
#concrete for filling
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    
    def __init__(self, method = "mean", fill_value = None):
        
        self.method = method
        self.fill_value = fill_value
        
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"filling missing values using method:{self.method} and fill value = {self.fill_value}")
        
        df_cleaned = df.copy()
        if self.method == "mean":
            
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            
            for column in df_cleaned.columns:
                
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace= True)
        elif self.method == "constant":
            
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            
            logging.warning(f"Unknown method '{self.method}'. No missing vlaues handled")
        
        logging.info("Missing vlaues filled")
        return df_cleaned        
    
#contxt class
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        
        self._strategy = strategy
        
    def set_strategy(self, strategy:MissingValueHandlingStrategy):
        
        logging.info("Switching missing value strategy")
        self._strategy = strategy
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info("Executing strategy for handling missing values.")
        return self._strategy.handle(df)
    
if __name__ == "__main__":
    pass