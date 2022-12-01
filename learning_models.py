# These are the machine learning models to run on the processed data file.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from operator import add
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import f1_score, accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import classification_report


def read_processed_data_file():
    # Read pre-processed data file into a dataframe
    path = './data/'
    name = 'processed-nfl-data.csv'
    df = pd.read_csv(path + name, index_col=0)
    return df


def model_choice():
    # User chooses which ML model to run
    choice = input("Select a model number (1=SVM ,2=Naive Bayes, 3=Decision Tree, 4=Random Forest): ")
    model_name = "quit"
    if choice == '1':
        model_name = "SVM"
    if choice == '2':
        model_name = "Naive Bayes"
    if choice == '3':
        model_name = "Decision Tree"
    if choice == '4':
        model_name = "Random Forest"

    print("You chose: ", choice, model_name)
    return model_name


def construct_model(kfold_choice):
    # Parse data and run selected machine learning model
    input_data = read_processed_data_file()     # get the processed data
    X, y = split_xy(input_data)  # split into training and test sets

    choice = model_choice()     # choose model type
    while choice != 'quit':

        if kfold_choice != "y":  # if not using 10-fold cross validation
            split = 0.20  # 20% of data used as test, 80% used for training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=None)
            print("construct_model: Data successfully split.")
            print("X_train:", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)
            scores = run_model(choice, kfold_choice, X_train, X_test, y_train, y_test)  # train selected model

        else:  # if using 10-fold cross validation
            folds = 10
            k_fold = KFold(folds, shuffle=True)
            count = 1
            sums = [0, 0]
            for train, test in k_fold.split(X, y):
                print(f"Fold number: {count}")
                X_train = X.iloc[train]
                X_test = X.iloc[test]
                y_train = y.iloc[train]
                y_test = y.iloc[test]
                scores = run_model(choice, kfold_choice, X_train, X_test, y_train, y_test) # train model
                sums = list(map(add, sums, scores))
                count += 1

            print(f"Average of {folds} Accuracy Scores: {sums[0] / folds}")
            print(f"Average of {folds} F1 Scores: {sums[1] / folds}")

        choice = model_choice()

    return True


def run_model(model_name, kfold_choice, X_train, X_test, y_train, y_test):
    # Train a model with selected machine learning algorithm
    if model_name == "SVM":
        model = SVC(C=10, gamma=0.01)
    if model_name == "Naive Bayes":
        model = GaussianNB()
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    if model_name == "Random Forest":
        model = RandomForestClassifier()

    starttime = datetime.now()
    model.fit(X_train, y_train)
    if kfold_choice != "y":
        print("run_model: ", model_name, " model built.")
        endtime = datetime.now()
        print("Run time to construct model: ", endtime - starttime)

        yes = input("Perfect this model? (y/n)")
        if yes == "y":
            perfect_model(model_name, model, X_train, y_train)
        print(model.get_params())
        y_train_pred = model.predict(X_train)
        print("Explained Variance Score (training): ", explained_variance_score(y_train, y_train_pred))

        save_model(model_name, model)

    y_pred = model.predict(X_test)
    scores = analyze_model(y_test, y_pred, kfold_choice)

    return scores


def split_xy(df):
    # split out classification (y) data
    y_raw = df['passResult']
    x_raw = df.drop(labels=['passResult'], axis=1)
    X = impute_values(x_raw)  # impute values that are NaN

    #lb = LabelBinarizer()  NOT USED
    #outcomes = ['C', 'I', 'S', 'R', 'IN']
    #y_array = lb.fit_transform(y_raw)
    #y = pd.DataFrame(y_array, columns=[outcomes[i] for i in range(len(outcomes))])

    play_map = {"C": 1, "I": 0, "S": 0, "IN": 0, "R": 0}
    y = y_raw.replace(play_map)  # Make categories binary
    print("Play Counts: ")
    print(y.value_counts())
    return X, y


def impute_values(df):
    # Replaces NaN values with K-Nearest Neighbor values
    imputer = KNNImputer(n_neighbors=2, weights="uniform").fit_transform(df)
    imputed = pd.DataFrame(imputer, columns=df.columns)  # turn back into a dataframe
    return imputed


def select_features(X_train, X_test):
    # Used for feature selection step; ultimately discarded
    print("Correlations: ")
    features = SelectPercentile(score_func=f_regression, percentile = 33)  # select bast features
    X_selected = features.fit(X_train, y_train)
    x = X_selected.get_support(indices=True)
    column_names = list(X_train.columns.values)
    return column_names


def perfect_model(model_name, model, X_train, y_train):
    # Use grid search to find optimized parameters
    if model_name == "SVM":
        params = [{'C': [0.001, 0.01, 0.1, 1, 10],
                   'gamma': [0.001, 0.01, 0.1, 1]}]
    if model_name == "Random Forest":
        params = [{'max_depth': [20, 50, 100],
                   'max_features': ['log2', 'sqrt', None],
                   'min_samples_split': [2, 5, 8],
                   'n_estimators': [100, 200, 400, 800]}]
    grid_search = GridSearchCV(model, params, cv=3, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("The optimized parameters are: ")
    print(grid_search.best_params_)
    return True


def analyze_model(y_test, y_pred, kfold):
    # tests classification models
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy}, F1 score: {f1}")
    if kfold != 'y':
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    return [accuracy, f1]


def save_model(choice, model):
    path = './model/'
    filename = choice+"_saved_model.sav"
    yes = input("Save this model? (y/n)")
    if yes == "y":
        pickle.dump(model, open(path + filename, 'wb'))
        print(f"{choice}: Model has been saved as {choice}_saved_model.sav")
    return True
