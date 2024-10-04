from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

class BivariateAnalysisStrategy(ABC):
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        
        pass
    
#concrete class

class NumericalNumericalAnalysis(BivariateAnalysisStrategy):
    
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x = feature1, y = feature2, data= df)
        plt.title(f"{feature1} v/s {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show
        
class CatergoricalNumericalAnalysis(BivariateAnalysisStrategy):
    
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
                
        plt.figure(figsize=(10, 6))
        sns.boxplot(x = feature1, y = feature2, data= df)
        plt.title(f"{feature1} v/s {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation = 45)
        plt.show
        
#main analyzer
class BivariateAnalyzer:
    
    def __init__(self, strategy: BivariateAnalysisStrategy):
        
        self._strategy = strategy
        
    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        
        self._strategy = strategy
        
    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        
        self._strategy.analyze(df, feature1, feature2)
        

#client code 

if __name__ == "__main__":
    pass