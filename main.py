# This is the main Python script. It provides the user interface to run subroutines.
from data_transformation import prepare_input
from learning_models import construct_model
from model_prediction import predict_outcome


# Runs the script.
if __name__ == '__main__':

    input_choice = input("Do you need to transform the NFL dataset? (y/n) ")  # process raw data?
    if input_choice == "y":
        prepare_input()  # load and prepare raw data
    else:
        print("File name is 'nfl-processed'.")

    training_choice = input("Do you want to train a new model? (y/n) ")  # train model on data?
    if training_choice == "y":
        kfold_choice = input("Do you want to use 10-fold cross validation? (y/n) ")  # use cross validation?
        construct_model(kfold_choice)

    output_choice = input("Do you want to apply a model to data? (y/n) ")  # apply a saved model?
    if output_choice == "y":
        predict_outcome()

    print("That's all, Folks!")
