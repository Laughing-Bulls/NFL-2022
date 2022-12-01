# This program uses saved models to supply predicted answers given provided inputs.
import numpy as np
import pandas as pd
import pickle
from data_transformation import transform
from data_transformation import scale_values
from data_transformation import label_code
from learning_models import split_xy


def read_play_data():
    path = './data/'
    name = 'new_play_inputs.csv'
    df = pd.read_csv(path + name, header=0)
    return df


def read_ml_models():
    path = './model/'
    filename1 = 'SVM_saved_model.sav'
    filename2 = 'Random Forest_saved_model.sav'
    saved_model1 = pickle.load(open(path + filename1, 'rb'))
    saved_model2 = pickle.load(open(path + filename2, 'rb'))
    return saved_model1, saved_model2


def predict_outcome():
    # Calculates the predicted score, based on saved ML models
    SVMmodel, RFmodel = read_ml_models()
    print("predict_outcome: The model outputs are from the currently saved models.")
    model_inputs = play_data()
    count = 1
    for index, row in model_inputs:
        print(index)
        print(count)
        print(row)

        prediction1 = 0
        prediction2 = 0
        #prediction1 = model1.predict(model_inputs)
        if prediction1 == 1:
            outcome1 = "SUCCESSFUL PLAY"
        else:
            outcome1 = "Unsuccessful play"
        print(f"For play {count}, the SVM model predicts: {outcome1}")

        #prediction2 = model1.predict(model_inputs)
        if prediction2 == 1:
            outcome2 = "SUCCESSFUL PLAY"
        else:
            outcome2 = "Unsuccessful play"
        print(f"For play {count}, the Random Forest model predicts: {outcome2}")

        actual = model_inputs.at[count, 'passResult']
        print(f"The actual outcome was: {actual}")
        print()

        count += 1

    return True


def play_data():
    # obtains inputted plays to evaluate and processes the data
    plays = read_play_data()
    processed_plays = transform(plays)
    X, y = split_xy(processed_plays)
    print("predict_outcome: The actual outcomes were: ", plays['passResult'])
    print(X)
    return X
