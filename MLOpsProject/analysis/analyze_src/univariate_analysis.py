from abc import ABC, abstractmethod

import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 


#step1 creating a abstract class

class UnivariateAnalysisStrategy(ABC):
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature:str):
        
        pass
    
#step2 concrete class
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    
    def analyze(self, df: pd.DataFrame, feature: str):
        
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], kde=True, bins=50)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()
        
#similarly for categorical

class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    
    def analyze(self, df: pd.DataFrame, feature: str):
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()
        
#step3 main class of analyzer

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy
        
    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self._strategy.analyze(df,feature)
        

#step4 client code

if __name__ == "__main__":
    pass