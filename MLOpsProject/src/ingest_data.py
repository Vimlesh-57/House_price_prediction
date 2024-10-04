import os
import zipfile
import csv
from abc import ABC, abstractmethod
import pandas as pd

#abstract data ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path:str) -> pd.DataFrame:
        pass

#implementing concrete zip ingestion class

class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path:str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("not a zip file")
        
        extract_dir = r"MLOpsProject\extracted_data"
        
        with zipfile.ZipFile(file_path,"r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        #extracted csv file
        extracted_files = os.listdir(extract_dir)
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        
        if len(csv_files) == 0:
            raise ValueError("no csv files.")
        if len(csv_files) > 1:
            raise ValueError("multiple csv file. specify the particular file")
        
        #read csv in dataframe
        csv_file_path = os.path.join(extract_dir, csv_files[0])
        df = pd.read_csv(csv_file_path)
        
        return df
    
    
    
#implementing factory
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension:str) -> DataIngestor:
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"no ingestor available for {file_extension}")
        

if __name__ == "__main__":
    
    file_path = r"C:\Users\vimle\Desktop\Complete MLOps\MLOpsProject\Data\Archive.zip"
    
    file_extension = os.path.splitext(file_path)[1]
    
    dataIngestor = DataIngestorFactory.get_data_ingestor(file_extension)
    
    df = dataIngestor.ingest(file_path)
    
    print(df.head())
    pass    