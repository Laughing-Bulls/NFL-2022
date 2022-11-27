# These are the machine learning models to run on the processed data file.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score


def read_processed_data_file():
    path = './data/'
    name = 'processed-nfl-data.csv'
    df = pd.read_csv(path + name, index_col=0)
    return df


def model_choice():
    choice = input("Select a model number (1,2,3,4,5,6,7,8,9,10): ")  # user can choose ML model
    model_name = "quit"
    if choice == '1':
        model_name = "Linear Regression"
    if choice == '2':
        model_name = "Lasso Regression"
    if choice == '3':
        model_name = "Stochastic Gradient Descent"
    if choice == '4':
        model_name = "Baysian Ridge Regression"
    if choice == '5':
        model_name = "Ridge Regression"
    if choice == '6':
        model_name = "Linear SVR"
    if choice == '7':
        model_name = "KNN Regression"
    if choice == '8':
        model_name = "Decision Tree Regression"
    if choice == '9':
        model_name = "Random Forest Regression"
    if choice == '10':
        model_name = "Gradient Boosting Regression"
    print("You chose: ", choice, model_name)
    return model_name


def construct_model():
    # parse data and run selected machine learning model
    input_data = read_processed_data_file()     # get the processed data

    X_train, X_test, y_train, y_test = split_data(input_data)   # split into training and test sets

    choice = model_choice()     # choose model type
    while choice != 'quit':
        model = run_model(choice, X_train, X_test, y_train, y_test)     # train selected model
        choice = model_choice()
    return True


def run_model(model_name, X_train, X_test, y_train, y_test):
    # Train a model with selected machine learning algorithm
    weights = True
    if model_name == "Linear Regression":
        model = LinearRegression()
    if model_name == "Lasso Regression":
        model = Lasso(alpha=1.0)
    if model_name == "Stochastic Gradient Descent":  #KEEP
        model = SGDRegressor(max_iter=1000, tol=1e-3)
    if model_name == "Baysian Ridge Regression":
        model = BayesianRidge()
    if model_name == "Ridge Regression":
        model = Ridge(alpha=0.5, tol=0.01)
    if model_name == "Linear SVR":
        model = LinearSVR()
    if model_name == "KNN Regression":
        model = KNeighborsRegressor()
    if model_name == "Decision Tree Regression":
        model = tree.DecisionTreeRegressor(min_samples_leaf=2, splitter='random')
        weights = False
    if model_name == "Random Forest Regression":
        model = RandomForestRegressor()
        weights = False
    if model_name == "Gradient Boosting Regression":
        model = GradientBoostingRegressor()
        weights = False

    starttime = datetime.now()
    model.fit(X_train, y_train)
    print("ML_Models: ", model_name, " model built.")
    endtime = datetime.now()
    print("Run time to construct model: ", endtime - starttime)
    yes = input("Perfect this model? (y/n)")
    if yes == "y":
        perfect_model(model_name, model, X_train, y_train)
    print(model.get_params())
    y_train_pred = model.predict(X_train)
    print("Explained Variance Score (training): ", explained_variance_score(y_train, y_train_pred))
    y_pred = model.predict(X_test)
    analyze_model(y_test, y_pred)
    save_model(model_name, model, weights)
    return True


def split_data(df):
    # split data into training and test data (80% / 20%)
    y_raw = df['passResult']
    x_raw = df.drop(labels=['passResult'], axis=1)
    X = impute_values(x_raw)  # impute values that are NaN

    lb = LabelBinarizer()
    cats = ['C', 'I', 'S', 'R', 'IN']
    y_array = lb.fit_transform(y_raw)
    y = pd.DataFrame(y_array, columns=[cats[i] for i in range(len(cats))])
    print(type(y))
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    #X_train = X.iloc[33:263351]
    #X_test = X.iloc[263382:350898]
    #y_train = y.iloc[33:263351]
    #y_test = y.iloc[263382:350898]

    print("ML_Models: Data successfully split.")
    print("X_train:", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)
    print(y_train.head())

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
    # tests regression models
    #print("Model score: ", model.score(X_test, y_test))
    print("R-squared: ", r2_score(y_test, y_pred))
    print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
    print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Explained Variance Score (test): ", explained_variance_score(y_test, y_pred))
    runplot(y_test, y_pred)
    return True


def runplot(y_test, y_pred):
    # plots graph of actual vs. predicted
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test, edgecolors=(0, 0, 1))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()
    return True


def save_model(choice, model, weights):
    path = './model/'
    filename = "learning_model.sav"
    parameter_filename = "parameters.csv"
    yes = input("Save this model? (y/n)")
    if yes == "y":
        pickle.dump(model, open(path + filename, 'wb'))
        print(choice, " : Model has been saved as final_model.sav")
        if weights:
            model_weights = model.coef_
            np.savetxt(path + parameter_filename, model_weights, delimiter=',')
            print("Model weights/coefficients have been saved as parameters.csv")
    return True
