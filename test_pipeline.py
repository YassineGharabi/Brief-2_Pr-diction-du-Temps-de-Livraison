import pytest
import pandas as pd
from pipeline import model_evaluation , data_cleaning , data_ecoding , data_scaling , get_features_target , split_data , train_model_grid_search 

@pytest.fixture
def load_data():
    return pd.read_csv('Data.csv')


def test_shape(load_data):
    assert load_data.shape == (1000, 9)


def test_evaluation(load_data):
    cleaned_data = data_cleaning(load_data)
    encoded_data = data_ecoding(cleaned_data)
    scaled_data = data_scaling(encoded_data)
    x_y_data = get_features_target(scaled_data)
    splited_data = split_data(x_y_data["x_features"],x_y_data["y_target"])
    trained_model = train_model_grid_search(splited_data["x_train"],splited_data["y_train"])
    scores = model_evaluation(trained_model,splited_data["x_test"],splited_data["y_test"])
    assert scores["mae"] <= 7
