import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pickle

def save_model(model, scaler,   file_path):
    with open(file_path, 'wb') as file:
        pickle.dump((model, scaler), file)


def generate_model():
    data = pd.read_csv("dataset.csv")

    # Convert string values to numerical values
    data['fight_result'] = data['fight_result'].map({'NOT_OVER': 0, 'P1': 1, 'P2': 2})

    # Convert boolean values to integer values
    bool_columns = ["p1_is_crouching", "p1_is_jumping", "p1_is_player_in_move", "p1_up","p1_down", "p1_right", "p1_left", "p1_select", "p1_start", "p1_Y", "p1_B", "p1_X", "p1_A", "p1_L", "p1_R", "p2_is_crouching", "p2_is_jumping", "p2_is_player_in_move", "p2_up", "p2_down", "p2_right" ,"p2_left", "p2_select", "p2_start", "p2_Y", "p2_B", "p2_X", "p2_A", "p2_L", "p2_R", "has_round_started", "is_round_over"]
    for column in bool_columns:
        data[column] = abs(data[column].astype(int))

    # Filter out rows where all p1 buttons are false
    p1_buttons = ["p1_up", "p1_down", "p1_right", "p1_left", "p1_Y", "p1_B", "p1_X", "p1_A", "p1_L", "p1_R"]
    data = data[data[p1_buttons].any(axis=1)]

    print(len(data['p1_down']))
    # Prepare input features and target variables
    input_features = data.drop(["p1_player_id", "p2_player_id","p1_up", "p1_down", "p1_right", "p1_left", "p1_Y", "p1_B", "p1_X", "p1_A", "p1_L", "p1_R", "p2_up", "p2_down", "p2_right" ,"p2_left", "p2_Y", "p2_B", "p2_X", "p2_A", "p2_L", "p2_R"], axis=1)
    
    target = data[["p1_up", "p1_down", "p1_right", "p1_left", "p1_Y", "p1_B", "p1_X", "p1_A", "p1_L", "p1_R"]]

    #nan_rows = data[data.isnull().any(axis=1)]
    #print(nan_rows)

    scaler = StandardScaler()
    input_features = scaler.fit_transform(input_features)

    X_train, X_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2, random_state=42)

    clf = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test R-squared: {r2}")
    print(f"Test Mean Absolute Error: {mae}")
    print(f"Test Mean Squared Error: {mse}")

    save_model(clf, scaler, "trained_model.pkl")



def load_model(file_path):
    with open(file_path, 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler


def make_prediction(model, scaler, input_data_df):

    input_data_df['fight_result'] = input_data_df['fight_result'].map({'NOT_OVER': 0, 'P1': 1, 'P2': 2})

    bool_input_columns = ["p1_is_crouching", "p1_is_jumping", "p1_is_player_in_move", "p1_up","p1_down", "p1_right", "p1_left", "p1_select", "p1_start", "p1_Y", "p1_B", "p1_X", "p1_A", "p1_L", "p1_R", "p2_is_crouching", "p2_is_jumping", "p2_is_player_in_move", "p2_up", "p2_down", "p2_right" ,"p2_left", "p2_select", "p2_start", "p2_Y", "p2_B", "p2_X", "p2_A", "p2_L", "p2_R", "has_round_started", "is_round_over"]
    for column in bool_input_columns:
        if column in input_data_df.columns:
            input_data_df[column] = input_data_df[column].astype(int)

    input_features = input_data_df.drop(["p1_player_id", "p2_player_id", "p1_up", "p1_down", "p1_right", "p1_left", "p1_Y", "p1_B", "p1_X", "p1_A", "p1_L", "p1_R", "p2_up", "p2_down", "p2_right" ,"p2_left", "p2_Y", "p2_B", "p2_X", "p2_A", "p2_L", "p2_R"], axis=1)

    input_features = scaler.transform(input_features)

    prediction = model.predict(input_features)
    prediction = np.where(prediction > 0.5, 1, 0)

    return prediction
