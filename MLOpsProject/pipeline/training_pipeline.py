from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_steps import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_handling_step import outlier_handling_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from zenml import Model, pipeline, step


@pipeline(
    model  =Model(
        name = "House_price_predictor"
    ),
)
def ml_pipeline():
    
    #data ingestion
    raw_data = data_ingestion_step(
        file_path = r"C:\Users\vimle\Desktop\Complete MLOps\MLOpsProject\Data\Archive.zip"
    )
    
    #handling missing values
    filled_data = handle_missing_values_step(raw_data)
    
    #feature engineering 
    engineered_data = feature_engineering_step(filled_data, strategy = "log", feature = ["Gr Liv Area", "SalePrice"])
    
    #outlier handling
    clean_data = outlier_handling_step(engineered_data, column_name = 'SalePrice')
    
    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column='SalePrice')
    
    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)
    
    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    run = ml_pipline()
    