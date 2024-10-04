import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path:str) -> pd.DataFrame:
    file_ext = ".zip"
    
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_ext)
    
    df = data_ingestor.ingest(file_path)
    return df