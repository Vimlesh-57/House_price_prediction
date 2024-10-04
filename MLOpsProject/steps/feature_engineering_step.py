import pandas as pd 
from src.feature_engineering import (FeatureEngineering, 
                                     LogTransformation, 
                                     MinMaxScaling, 
                                     StandardTransformation, 
                                     OneHotEncoding,
                                     )
from zenml import step

@step
def feature_engineering_step(df: pd.DataFrame, strategy: str = "log", feature: list = None) -> pd.DataFrame:
    
    if feature is None:
        
        feature = []
        
    if strategy == "log":
        engineer = FeatureEngineering(LogTransformation(feature))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineering(StandardTransformation(feature))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineering(MinMaxScaling(feature))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineering(OneHotEncoding(feature))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df
