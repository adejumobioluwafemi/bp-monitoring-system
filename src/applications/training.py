from ml.training_pipeline import TrainingPipeline
from data_layer.data_connector import DataConnector
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def run():
    # Get data
    df = DataConnector.get_data(
        'bp-monitoring-system/data/Hypertension_Data _Set.csv')
    X = df.drop(columns=['Hypertension_Tests'])
    y = df['Hypertension_Tests']

    # Define columns
    numerical_cols = ['Age', 'BMI', 'Systolic_BP',
                      'Diastolic_BP', 'Heart_Rate']
    categorical_cols = ['Gender', 'Medical_History',
                        'Smoking', 'Sporting']

    # Initialize and run pipeline
    pipeline = TrainingPipeline(numerical_cols, categorical_cols)
    pipeline.fit(X, y)

    # Save the trained model
    pipeline.get_model().save('models_chpt/hypertension_model.pkl')


if __name__ == '__main__':
    run()
