from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

#step1 base class for missing values
class MissingValueAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
        
    @abstractmethod
    def identify_missing_values(self, df:pd.DataFrame):
        pass
    
    @abstractmethod
    def visualize_missing_values(self, df:pd.DataFrame):
        pass
    
#step2 concrete class for missing value identification
class SimpleMissingValuesAnalysis(MissingValueAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        print("\n Missing values counted by columns:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values>0])
        
    def visualize_missing_values(self, df: pd.DataFrame):
        print("\n Visualizing missing values...")
        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull(), cbar= False, cmap="viridis")
        plt.title("Heatmap for missing values")
        plt.show()
        
#step 3 client code
if __name__ == "__main__":
    #example usage 
    #load the data
    #df = pd.read_csv("file_path")
    
    #perform missing value analysis
    #missing_value_analyzer = SimpleMissingValuesAnalysis()
    #missing_values_analyzer.analyze(df)
    
    pass
