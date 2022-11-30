# These programs take the raw NFL data as a dataframe and create a pre-processed data file.
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer, OneHotEncoder


def read_data():
    path = './data/'
    name = 'plays.csv'
    #df = pd.read_csv(path + name, dtype={'gameClock':str}, header=0)
    df = pd.read_csv(path + name, header=0)

    return df


def write_data(df):
    path = './data/'
    name = 'processed-nfl-data.csv'
    df.to_csv(path + name)
    return True


def prepare_input():
    # Calls functions to read, process and transform raw data, and save it as csv
    raw_data = read_data()
    visualize_data(raw_data)
    play_input = transform(raw_data)    # Call function to transform the data
    write_data(play_input)
    return True


def transform(df):
    # Transforms the raw data.
    # df.drop(index=df.index[0:8000], inplace=True)  # make data set smaller for testing
    field_side = 1 - (df['defensiveTeam'] == df['yardlineSide'])  # 1 if not in defensive territory
    df['yardsToEndzone'] = abs(df['yardlineNumber'] - field_side * 100)  # modify feature for side of field
    df['firstHalf'] = (df['quarter'] < 3)   # adjust quarters to halves
    df['overtime'] = (df['quarter'] > 4)    # overtime feature

    addTime = ((df['quarter'] == 1) | (df['quarter'] == 3))   # adjust game clock to time left in half
    time = df['gameClock']
    nums = time.str.partition(':')
    nums[0] = pd.to_numeric(nums[0])
    nums[2] = pd.to_numeric(nums[2])
    df['timeLeft'] = nums[0] + nums[2]/60 + (addTime * 15)

    df['scoreDifference'] = abs(df['preSnapHomeScore'] - df['preSnapVisitorScore'])  # score differential

    df.drop(labels=['gameId', 'playId', 'playDescription', 'quarter', 'possessionTeam', 'defensiveTeam',
                    'yardlineSide', 'yardlineNumber', 'gameClock', 'preSnapHomeScore', 'preSnapVisitorScore',
                    'prePenaltyPlayResult', 'penaltyYards', 'playResult', 'foulName1', 'foulNFLId1', 'foulName2',
                    'foulNFLId2', 'foulName3', 'foulNFLId3', 'absoluteYardlineNumber'],
            axis=1, inplace=True)  # remove less relevant column

    scaled = scale_values(df)   # function returns min-max feature scaling on numerical values
    print("transform: The transformed NFL play columns are: ")
    print(df.columns)   # show transformed data headers
    coded = label_code(scaled)  # function returns transformed array

    return coded


def scale_values(df):
    # scale numerical data
    scaler = MinMaxScaler()
    #column_heads = ['down', 'yardsToGo', 'defendersInBox', 'yardsToEndzone', 'timeLeft', 'scoreDifference']
    df[['down', 'yardsToGo', 'defendersInBox', 'yardsToEndzone', 'timeLeft', 'scoreDifference']] \
        = pd.DataFrame(scaler.fit_transform(df[['down', 'yardsToGo', 'defendersInBox', 'yardsToEndzone',
                                                'timeLeft', 'scoreDifference']]))
    return df


def label_code(df):
    # transform categories (e.g., venues, teams) into categorical integers with value 0 or 1
    coded = pd.get_dummies(data=df, columns=['offenseFormation', 'personnelO', 'personnelD',
                                             'dropBackType', 'pff_passCoverage', 'pff_passCoverageType'])
    print('label_code: The shape of the transformed dataframe is:', coded.shape)
    return coded


def visualize_data(df):
    # description and graphical representation of raw data
    print("visuaize_data: Raw data: ")
    print(df.head())  # describe raw data
    print(df.shape)
    print(df.describe(include='all'))
    print('visuaize_data: The raw data headers are:')
    print(df.columns)   # show raw data headers
    df['passResult'].hist(color="palevioletred", edgecolor="black")
    plt.show()      # histogram of play outcomes
    figures = plt.figure()
    axis = figures.gca()
    df[['down', 'defendersInBox', 'prePenaltyPlayResult',
        'yardlineNumber']].hist(ax=axis, color="palevioletred", edgecolor="black")
    plt.show()      # histograms of various input variables
    return True
