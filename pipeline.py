import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error , r2_score



def load_data():
    return pd.read_csv('Data.csv')


def data_cleaning(data):
    df = data.copy()
    return df.fillna({
    'Weather': data['Weather'].mode()[0],
    'Traffic_Level': data['Traffic_Level'].mode()[0],
    'Time_of_Day': data['Time_of_Day'].mode()[0],
    'Courier_Experience_yrs': data['Courier_Experience_yrs'].median()
    })

def data_ecoding(data):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_cols = []
    for column in data.columns:
        if( data[column].dtype == 'O' ):
            cat_cols.append(column)


    encoded = encoder.fit_transform(data[cat_cols])

    encoded_data = pd.DataFrame( encoded , columns=encoder.get_feature_names_out(cat_cols) )


    # Supprimer anciennes colonnes catégorielles
    data = data.drop(cat_cols, axis=1)

    # Fusionner avec le DataFrame principal
    data = pd.concat([data, encoded_data], axis=1)
    return data

def data_scaling(data):
    scaler = StandardScaler()
    data[ [ "Distance_km" , "Preparation_Time_min" , "Courier_Experience_yrs"  ] ] = scaler.fit_transform(data[[ "Distance_km" , "Preparation_Time_min" , "Courier_Experience_yrs"  ]])
    return data


# get the best features with (k best) and taget
def get_features_target(data):
    x_features = data.drop("Delivery_Time_min", axis=1)
    y_target = data["Delivery_Time_min"]

    selector = SelectKBest(score_func=f_regression , k=8)

    selector.fit_transform( x_features , y_target ) 

    # Obtenir les noms des colonnes sélectionnées
    selected_mask = selector.get_support()  # Boolean mask
    selected_features = x_features.columns[selected_mask]



    # change x_features to the selected
    x_features = data[selected_features.to_list()]

    return { "x_features" : x_features , "y_target" : y_target }


def split_data(x_features,y_target):
    x_train , x_test , y_train , y_test = train_test_split( x_features , y_target , random_state=42 , test_size=0.2  )
    return { "x_train" : x_train , "x_test" : x_test , "y_train" : y_train , "y_test" : y_test }


def train_model_grid_search(x_train,y_train):
    model_svr = SVR()

    # Define the parameter grid
    param_grid_svr = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],  
        "C": [0.1, 1, 10, 100],                          
        "gamma": ["scale", "auto"],                      # kernel coefficient
        "epsilon": [0.01, 0.1, 0.2, 0.5]                # margin of tolerance
    }

    # Define the GridSearchCv
    grid_search_svr = GridSearchCV(
        estimator=model_svr,
        param_grid=param_grid_svr,
        cv=5,
        # scoring='r2',
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    # Train the model
    grid_search_svr.fit( x_train , y_train )

    # Get the best parameters and score
    # print("Best parameters found : ", grid_search_svr.best_params_)
    # print("Best score : ", grid_search_svr.best_score_)

    # get the best model
    svr_best_model = grid_search_svr.best_estimator_
    return svr_best_model

def model_evaluation(model,x_test,y_test):
    svr_y_pred = model.predict( x_test )

    r2_svr = r2_score( y_test , svr_y_pred )
    mea_svr = mean_absolute_error( y_test , svr_y_pred )

    return { "r2_score" : r2_svr , "mae" : mea_svr }









def main():

    data = load_data()
    cleaned_data = data_cleaning(data)
    encoded_data = data_ecoding(cleaned_data)
    scaled_data = data_scaling(encoded_data)
    x_y_data = get_features_target(scaled_data)
    splited_data = split_data(x_y_data["x_features"],x_y_data["y_target"])
    trained_model = train_model_grid_search(splited_data["x_train"],splited_data["y_train"])
    scores = model_evaluation(trained_model,splited_data["x_test"],splited_data["y_test"])
    print(scores)

main()