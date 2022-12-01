# This program uses saved ML models to supply predictions, given provided inputs.
import numpy as np
import pandas as pd
import pickle
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
    SVMmodel, RFmodel = read_ml_models()    # get saved models
    print("predict_outcome: The model outputs are from the currently saved models.")
    model_inputs, outcomes = play_data()
    predictions1 = SVMmodel.predict(model_inputs)   # generate SVM predictions
    predictions2 = RFmodel.predict(model_inputs)    # generate Random Forest predictions

    count1 = 0  # keep track of how many predictions each model gets right
    count2 = 0
    for index in range(len(model_inputs)):  # iterate through predictions
        actual = outcomes[index]
        prediction1 = predictions1[index]
        prediction2 = predictions2[index]

        if prediction1 == 1:
            outcome1 = "SUCCESSFUL PLAY"
        else:
            outcome1 = "Unsuccessful play"

        print(f"For play {index}, the SVM model predicts: {prediction1} {outcome1}")
        if prediction1 == actual:
            count1 +=1

        if prediction2 == 1:
            outcome2 = "SUCCESSFUL PLAY"
        else:
            outcome2 = "Unsuccessful play"
        print(f"For play {index}, the Random Forest model predicts: {prediction2} {outcome2}")
        if prediction2 == actual:
            count2 +=1

        print(f"The actual outcome was: {actual}")
        print()

    print(f"Of {len(model_inputs)} plays, SVM got {count1} correct and Random Forest got {count2} correct")
    return True


def play_data():
    # Obtains inputted plays to evaluate
    plays = read_play_data()
    X, y = split_xy(plays)
    print("predict_outcome: The actual outcomes were: ")
    print(plays['passResult'])
    return X, y
