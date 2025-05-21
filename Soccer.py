# Installing the library

!pip install scikit-learn==1.0.2
!pip install lime
!pip install shap
!pip install tqdm

!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0TUZEN/lime.zip"
!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0TUZEN/shap.zip"

# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lime
import shap

%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from lime import lime_tabular
from itertools import accumulate

#Suppressing the warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Defining the helper functions
def plotter(x, y, title):
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

# Reading the data
!wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0TUZEN/Training_games.csv'
Game_all = pd.read_csv('Training_games.csv')
Game_all.head()

# List of teams
Team = pd.DataFrame({'team':['Senegal','Qatar','Netherlands','Ecuador','Iran','England','United States','Wales','Argentina','Saudi Arabia','Mexico','Poland','Denmark','Tunisia','France','Australia','Germany','Japan','Spain','Costa Rica','Morocco','Croatia','Belgium','Canada','Switzerland','Cameroon','Brazil','Serbia','Uruguay','Korea Republic','Portugal','Ghana']})
Teams = ['Senegal','Qatar','Netherlands','Ecuador','Iran','England','United States','Wales','Argentina','Saudi Arabia','Mexico','Poland','Denmark','Tunisia','France','Australia','Germany','Japan','Spain','Costa Rica','Morocco','Croatia','Belgium','Canada','Switzerland','Cameroon','Brazil','Serbia','Uruguay','Korea Republic','Portugal','Ghana']

# Replacing 'South Korea' to 'Korea Republic'
Game_all = Game_all.replace('South Korea','Korea Republic')
Game_all.head()

# Filter the teams which will go to World Cup 2022
Game_used = Game_all.drop(Game_all[(~Game_all['home_team'].isin(Teams)) | (~Game_all['away_team'].isin(Teams)) ].index)
Game_used.count()
Game_used

# Making sure every team going to the World Cup is in the training set
list3 = list(Game_used['home_team'].unique()) + list(set(Game_used['away_team'].unique()) - set(Game_used['home_team'].unique()))
list(set(list3) - set(Teams))

# The players for training
!wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0TUZEN/Player.csv'
Player_all = pd.read_csv('Player.csv')
Player_all.head()

# The number of players in each country
for i in range(len(Teams)):
    print(Teams[i],Player_all.loc[Player_all['Nationality'] == Teams[i]]['Name'].count())

# Looking at one player's data
Player_all.loc[Player_all.FullName == 'Kevin De Bruyne']

# Importing the prepared lineup
!wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0TUZEN/Training_game_62.csv'
Game_62 = pd.read_csv('Training_game_62.csv')

# Showing the first player
Player_all.loc[(Game_62.loc[0]['Player1'])]


# Training set


# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0TUZEN/Training_games.csv'



