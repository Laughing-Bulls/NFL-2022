# This is the MAIN Python script. It provides the user interface to run subroutines.
from transform_data import prepare_input
from ml_models import construct_model
from apply_model import predict_outcome


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


# Runs the script.
if __name__ == '__main__':

    input_choice = input("Do you need to transform the NFL dataset? (y/n)")  # process raw data?
    if input_choice == "y":
        prepare_input()  # load and prepare raw data
    else:
        print("File name is 'nfl-processed'.")

    training_choice = input("Do you want to train a new model? (y/n)")  # train model on data?
    if training_choice == "y":
        choice = model_choice()
        while choice != 'quit':
            model = construct_model(choice)  # run and evaluate selected machine learning model
            choice = model_choice()

    output_choice = input("Do you want to apply the model to data? (y/n)")  # apply the saved model?
    if output_choice == "y":
        predict_outcome()

    print("That's all, Folks!")
