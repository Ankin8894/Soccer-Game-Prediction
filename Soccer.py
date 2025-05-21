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
Team = pd.DataFrame({'team':['Senegal', 'Qatar', 'Netherlands', 'Ecuador', 'Iran', 'England', 'United States', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland', 'Denmark', 'Tunisia', 'France', 'Australia', 'Germany', 'Japan', 'Spain', 'Costa Rica', 'Morocco', 'Croatia', 'Belgium', 'Canada', 'Switzerland', 'Cameroon', 'Brazil', 'Serbia', 'Uruguay', 'Korea Republic', 'Portugal', 'Ghana']})
Teams = ['Senegal', 'Qatar', 'Netherlands', 'Ecuador', 'Iran', 'England', 'United States', 'Wales', 'Argentina', 'Saudi Arabia', 'Mexico', 'Poland', 'Denmark', 'Tunisia', 'France', 'Australia', 'Germany', 'Japan', 'Spain', 'Costa Rica', 'Morocco', 'Croatia', 'Belgium', 'Canada', 'Switzerland', 'Cameroon', 'Brazil', 'Serbia', 'Uruguay', 'Korea Republic', 'Portugal', 'Ghana']

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
team_age = []
for j in range(62):
    
    # Players from teams 1 and 2
    team1_index = list(Game_62.loc[j][8:19])
    team2_index = list(Game_62.loc[j][18:30])
    
    # Team age average
    temp1 = Player_all.iloc[team1_index]['Age'].mean()
    temp2 = Player_all.iloc[team2_index]['Age'].mean()
    team_age.append([temp1, temp2])

# Creating the DataFrame for the average age
cols = ['Average_age1', 'Average_age2']
Train_df_age = pd.DataFrame(team_age, columns = cols)
Train_df_age.head()

# DataFrame for every teams' player overall rating
team_overall = []
for j in range(62):

    # Players 1 and 2 index fromt he Team DataFrame
    team_index = list(Game_62.loc[j][8:30])
    
    # team1 and team2 player overall from the Team DataFrame
    temp3 = Player_all.iloc[team_index]['Overall']
    team_overall.append(list(temp3))

# Creating overall player's DataFrame
cols = []
for i in range(22):
    cols.append('Player'+str(i+1)+'_overall')
Train_df_overall = pd.DataFrame(team_overall, columns = cols)
Train_df_overall.head()

# Combining of the two DataFrames into Train_df
Train_df = pd.concat([Train_df_age, Train_df_overall], axis = 1)
Train_df.head()

# Setting up labels
Result_training = []
Result_totalgoal = []
Result_netgoal = []
for j in range(62):
    Result_training.append(Game_62.loc[j]['Result'])
    Result_totalgoal.append(Game_62.loc[j]['Netgoal'])
    Result_netgoal.append(Game_62.loc[j]['Goal_total'])
np.array(Result_training)
np.arrage(Result_totalgoal)
np.array(Result_netgoal)

# Importing all the group stage games:
!wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0TUZEN/matches_group.csv'
match = pd.read_csv('matches_group.csv')
match.head()

!wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0TUZEN/Teams.csv'
Team = pd.read_csv('Teams.csv')
Team.head()

# Preparing the predicting set
team_age = []
for j in range(48):

    # team1 and team2 player index from the Team DataFrame
    team1_index = list(Team.loc[Team['Team'] == match['country1'][j]].iloc[0, 3:])
    team2_index = list(Team.loc[Team['Team'] == match['country2'][j]].iloc[0, 3:])

    # Average of team age
    temp1 = Player_all.iloc[team1_index]['Age'].mean()
    temp2 = Player_all.iloc[team2_index]['Age'].mean()
    team_age.append([temp1, temp2])

# Creating the DataFrame for the average age
cols = ['Average_age1', 'Average_age2']
Predic_df_age = pd.DataFrame(team_age, columns = cols)
Predic_df_age.head()

# Team overall
team_overall = []
for j in range(48):

    # Full team1 and team2 player index from the Team DataFrame
    team_index = list(Team.loc[Team['Team'] == match['country1'][j]].iloc[0, 3:]) + list(Team.loc[Team['Team'] == match['country2'][j]].iloc[0, 3:])

    # team1 and team2 player overall from the Team DataFrame
    temp3 = Player_all.iloc[team_index]['Overall']
    team_overall.append(list(temp3))

# Creating the DataFrame for the player overall
cols = []
for i in range(22):
    cols.append('Player'+str(i+1)+'_overall')
Predic_df_overall = pd.DataFrame(team_overall, columns = cols)
Predic_df_overall.head()

# Combining the two DataFrames
Predicting_df = pd.concat([Predic_df_age, Predic_df_overall],axis = 1)
Predicting_df.head()

# Standardizing the predicting set and training set
All_df = pd.concat([Train_df, Predicting_df], axis = 0)
All_df = preprocessing.StandardScaler().fit(All_df).transform(All_df.astype(float))
X_1 = All_df[:62]
X_2 = All_df[62:]

# Now the training of the model
X_train, X_test, y_train, y_test = train_test_split(X_1, np.array(Result_training), test_size = 0.1, random_state = 42)
X_train.shape, X_test.shape
X_train

# Training the different parameters and finding the best one
parameters = {'hidden_layer_sizes':[50, 75, 100],
              'alpha': [0.0001, 0.001, 0.01, 0.1],
              'max_iter': [200, 500, 800],
              'learning_rate_init':[0.0001, 0.001, 0.01, 0.1]}

model = MLPClassifier()
clf = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)

print("The best parameter values found are:\n")
print(clf.best_params_)

# Storingthe best model found in "bestmodel"
bestmodel = clf.best_estimator_

# The testing result
y_prob = bestmodel.predict_proba(X_test)
y_pred = bestmodel.predict(X_test)
print(f"The accuracy score of the best model is {accuracy_score(y_test, y_pred)}\n")
print([np.round(y_prob,3), y_pred, y_test])

# Predicting the W/L/D and probability of them of each game
y_prob = bestmodel.predict_proba(X_2)
y_result = bestmodel.predict(X_2)
print([np.round(y_prob, 3), y_result])

# Combining the game predicting result with the time and Teams in one DataFrame
cols_result = ['Time', 'Home Team', 'Away Team', 'Predicted result', 'Home Win rate', 'Draw rate', 'Home Lose rate']
Result_df = pd.DataFrame([], columns = cols_result)
Result_df['Time'] = match.iloc[:48, 1].values
Result_df['Home Team'] = match.iloc[:48, 2].values
Result_df['Away Team'] = match.iloc[:48, 3].values
Result_df['Predicted result'] = y_result
Result_df['Home Win rate'] = np.round(y_prob[:,2], 3)
Result_df['Draw rate'] = np.round(y_prob[:,0], 3)
Result_df['Home Lose rate'] = np.round(y_prob[:,1], 3)
Result_df
content_result = np.concatenate(match.iloc[:,1].values, match.iloc[:,2].values, match.iloc[:,3].values, y_result, np.round(y_prob[:,2],3), np.round(y_prob[:,0],3), np.round(y_prob[:,1],3))
content_result.transpose()

Result_df

# Displaying in a diagram
A = Result_df.loc[Result_df['Home Team'] == 'Japan']
A = A.loc[A['Away Team'] == 'Spain']

data = {'Japan Win':A['Home Win rate'].values[0], 'Draw':A['Draw rate'].values[0], 'Spain Win':A['Home Lose rate'].values[0]}
number = list(data.keys())
values = list(data.values())
c = ['blue', 'yellow', 'red']

plt.bar(number, values, color = c, width = 0.4)
plt.title("Result probability of Japan VS Spain", y=1.012)
plt.ylabel("probability", labelpad=3);

plt.show()

# Using LIME to help with interpretability
Label_name = ['Team1_Age','Team2_Age','GK_1','RB_1','CB1_1','CB2_1','LB_1','M1_1','M2_1','M3_1','M4_1','F1_1','F2_1','GK_2','RB_2','CB1_2','CB2_2','LB_2','M1_2','M2_2','M3_2','M4_2','F1_2','F2_2']

# Setting the lime explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data = X_train,
    feature_names = Label_name,
    class_names = ['D','L','W'],
    mode = 'classification'
)

# Checking the game 28 Spain VS Germany
Exam_1 = X_2[27]

# Continuing to set the lime explainer
lime_exp = lime_explainer.explain_instance(
    data_row = Exam_1,
    predict_fn = bestmodel.predict_proba,
    num_features = 24
)

# Plotting the result
lime_exp.show_in_notebook(show_table = True)

# SHAP also helps with interpretability
# The SHAP explainer
explainer = shap.KernelExplainer(bestmodel.predict_proba, X_1)

#Calculating SHAP value
shap_values = explainer.shap_values(X_2)

# Plotting the features influence in general
shap.summary_plot(shap_values, X_2,
                  feature_names = Label_name,
                  class_names = ['D','L','W'],
                  max_display = 24)

shap.initjs() #initial javascript in cell
shap.force_plot(explainer.expected_value[0], shap_values[0], X_1, feature_names = Label_name)

# Showing for a specific player
Player_all.loc[Player_all.FullName == 'Lionel Messi']

# Displaying a team lineup in one game
Lineup_1 = []
for i in range(11):
    location = 'Player'+str(i+1)
    number = Game_62.loc[0][location]
    Lineup_1.append(Player_all.loc[number]['Ghana'])
Lineup_1

# Showing a specific game result prediction
B = Result_df.loc[Result_df['Home Team'] == 'Canada']
B = B.loc[B['Away Team'] == 'Morocco']

data = {'Canada Win':B['Home Win rate'].values[0], 'Draw':B['Draw rate'].values[0], 'Morocco Win':B['Home Lose rate'].values[0]}
number = list(data.keys())
values = list(data.values())
c = ['red', 'yellow', 'blue']

plt.bar(number, values, color = c, width = 0.4)
plt.title("Result probability of Canada VS Morocco", y = 1.012)
plt.ylabel("probability", labelpad = 3);

plt.show()
