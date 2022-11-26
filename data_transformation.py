# These programs take the raw NFL data as a dataframe and create a pre-processed data file.
import datetime
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
#import sklearn
#from sklearn.compose import ColumnTransformer
#from sklearn import preprocessing
#from sklearn.preprocessing import Normalizer, OneHotEncoder


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
    print("transform_data: Raw data: ")
    print(raw_data.head())  # Print raw data
    print(raw_data.shape)
    play_input = transform(raw_data)    # Call function to transform the data
    print("transform_data: The processed NFL game play inputs matrix: ")
    print(play_input.columns)
    print(play_input.head())  # Print transformed data
    print(raw_data.shape)
    write_data(play_input)
    return True


def transform(df):
    # Transforms the raw data.
    print('transform_data: The raw data headers are:')
    print(df.columns)
    df.drop(index=df.index[0:8000], inplace=True)  # make data set smaller for testing
    field_side = 1 - (df['defensiveTeam'] == df['yardlineSide'])  # 1 if not in defensive territory
    df['yardsToEndzone'] = abs(df['yardlineNumber'] - field_side * 100)  # modify feature for side of field

    # df['firstHalf'] = (df['quarter'] < 3)   # adjust quarters to halves
    df['overtime'] = (df['quarter'] > 4)    # overtime feature
    addTime = ((df['quarter'] == 1) | (df['quarter'] == 3))   # adjust game clock to time left in half

    time = df['gameClock']
    nums = time.str.partition(':')
    nums[0] = pd.to_numeric(nums[0])
    nums[2] = pd.to_numeric(nums[2])
    df['timeLeft'] = nums[0] + nums[2]/60 + (addTime * 15)

    df['scoreDifference'] = abs(df['preSnapHomeScore'] - df['preSnapVisitorScore'])  # score differential

    df.drop(labels=['gameId', 'playId', 'playDescription', 'quarter', 'possessionTeam', 'defensiveTeam',
                    'yardlineSide', 'yardlineNumber', 'gameClock', 'penaltyYards', 'playResult',
                    'foulName1', 'foulNFLId1', 'foulName2', 'foulNFLId2', 'foulName3', 'foulNFLId3',
                    'absoluteYardlineNumber'],
            axis=1, inplace=True)  # remove less relevant column

    coded = label_code(df)  # function returns transformed array
    # scaler = MinMaxScaler()  # do not scale
    # coded[coded.columns] = scaler.fit_transform(coded[coded.columns])  # scale to numerical data 0:1
    # print(coded.describe())
    return coded


def label_code(df):
    # transform categories (e.g., venues, teams) into categorical integers
    # convert each category into its own column with value 0 or 1
    coded = pd.get_dummies(data=df, columns=['passResult', 'offenseFormation', 'personnelO', 'personnelD',
                                             'dropBackType', 'pff_passCoverage', 'pff_passCoverageType'])
    #ct = ColumnTransformer([('encode', OneHotEncoder(categories='auto'))], remainder='passthrough') # column transformation
    #transformed = np.array(ct.fit_transform(df), dtype=np.str)
    #transformed_df = pd.DataFrame(transformed) # convert back to dataframe
    #le = preprocessing.LabelEncoder()
    #encoded = df.apply(le.fit_transform)  # each unique category assigned an integer
    #enc = preprocessing.OneHotEncoder()
    #enc.fit(encoded)
    #transformed = enc.transform(encoded).toarray()
    print('transform_data: The shape of the transformed array is:', coded.shape)
    return coded
