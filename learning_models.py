# These are the machine learning models to run on the processed data file.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from operator import add
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import f1_score, accuracy_score
from sklearn.metrics import explained_variance_score


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

    if kfold_choice == "y": # if using 10-fold cross validation
        num_runs = 10  # Train and test 10 times
        split = 0.10    # 10% of data used as test
    else:
        num_runs = 1    # Train and test once
        split = 0.20    # otherwise 20% of data used as test

    choice = model_choice()     # choose model type
    while choice != 'quit':
        sums = [0,0]
        for i in range(0, num_runs):
            print(f"Run number: {i+1}")
            X_train, X_test, y_train, y_test = split_data(input_data, split)  # split into training and test sets
            scores = run_model(choice, kfold_choice, X_train, X_test, y_train, y_test)     # train selected model
            sums = list(map(add, sums, scores))

        if kfold_choice == "y": # if using 10-fold cross validation
            print(f"Average of {num_runs} Accuracy Scores: {sums[0]/num_runs}")
            print(f"Average of {num_runs} F1 Scores: {sums[1] / num_runs}")

        choice = model_choice()

    return True


def run_model(model_name, kfold_choice, X_train, X_test, y_train, y_test):
    # Train a model with selected machine learning algorithm
    if model_name == "SVM":
        model = SVC()
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
    scores = analyze_model(y_test, y_pred)

    return scores


def split_data(df, test_size):
    # split data into training and test data (80% / 20% unless 10-fold is selected)
    y_raw = df['passResult']
    x_raw = df.drop(labels=['passResult'], axis=1)
    X = impute_values(x_raw)  # impute values that are NaN

    #lb = LabelBinarizer()
    #outcomes = ['C', 'I', 'S', 'R', 'IN']
    #y_array = lb.fit_transform(y_raw)
    #y = pd.DataFrame(y_array, columns=[outcomes[i] for i in range(len(outcomes))])

    play_map = {"C": 1, "I": 0, "S": 0, "IN": 0, "R": 0}
    y = y_raw.replace(play_map)  # Make categories binary
    print("Play Counts: ")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
    print("split_data: Data successfully split.")
    print("X_train:", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)

    return X_train, X_test, y_train, y_test


def impute_values(df):
    # Replaces NaN values with K-Nearest Neighbor values
    imputer = KNNImputer(n_neighbors=2, weights="uniform").fit_transform(df)
    imputed = pd.DataFrame(imputer, columns=df.columns)  # turn back into a dataframe
    return imputed


def select_features(X_train, X_test):

    print("Correlations: ")
    features = SelectPercentile(score_func=f_regression, percentile = 33)  # select bast features
    X_selected = features.fit(X_train, y_train)
    x = X_selected.get_support(indices=True)
    print(x)
    column_names = list(X_train.columns.values)
    print(column_names)
    for each in x:
        print(column_names[each])

    # X_selected = fs.fit_transform(X_train, y_train)
    # print(X_selected.shape)

    return X_train, X_test, y_train, y_test


def perfect_model(model_name, model, X_train, y_train):
    if model_name == "Ridge Regression":
        params = [{'alpha': [1e-1, 0.5, 1, 5, 10],
                   'copy_X': [True, False], 'tol': [0.01, 0.001, 0.0001]}]
    if model_name == "Decision Tree Regression":
        params = [{'splitter': ["best", "random"], 'max_depth': [1, 5, 8, 12],
                   'min_samples_leaf': [2, 5, 7, 10], 'min_weight_fraction_leaf': [0.1, 0.3, 0.5],
                   'max_features': ['auto', 'log2', 'sqrt', None],
                   'max_leaf_nodes': [None, 25, 50, 70, 90]}]
    grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=3,
                               return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("The optimized parameters are: ")
    print(grid_search.best_params_)
    return True

def analyze_model(y_test, y_pred):
    # tests classification models
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy}, F1 score: {f1}")

    #print("Model score: ", model.score(X_test, y_test))
    #print("R-squared: ", r2_score(y_test, y_pred))
    #print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
    #print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    #print("Explained Variance Score (test): ", explained_variance_score(y_test, y_pred))
    #runplot(y_test, y_pred)
    return [accuracy, f1]


def runplot(y_test, y_pred):
    # plots graph of actual vs. predicted
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test, edgecolors=(0, 0, 1))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()
    return True


def save_model(choice, model):
    path = './model/'
    filename = choice+"_saved_model.sav"
    yes = input("Save this model? (y/n)")
    if yes == "y":
        pickle.dump(model, open(path + filename, 'wb'))
        print(choice, " : Model has been saved as saved_model.sav")
    return True
