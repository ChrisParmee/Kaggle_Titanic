#! Python-3

#######################################################################
# Code for Titanic Kaggle competition submission, based off 
# notebooks for other submissions 
#######################################################################



###### - Import modules - #####
import numpy as np      
import pandas as pd     
import random as rnd    
import seaborn as sns   
import sys
import xgboost as xgb
import matplotlib.pyplot as plt 

from myFunctions import myPrint # Import my own print function
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 

## Import pyplot, but set block to false and have windows only appear for pause_time
def myPlot():
    plt.show(block=False)
    pause_time = 2
    plt.pause(pause_time)
    plt.close()


#######################################
##### - Import and analyse data - #####

### Import the data
gender_submission = pd.read_csv("gender_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
total = [train, test]

myPrint(total[0].head(), hold=False)

### Print the column values and last 10 data points for the training data
## Find columns are ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch', 'Ticket' 'Fare' 'Cabin' 'Embarked']
myPrint(train.columns.values, printmessage="Train Columns", hold=False)
myPrint(train.tail(10), printmessage="Train Tail", hold=False)

### Analyse the train and test data for missing values
# Find Age, Cabin and Embarked features have missing values for train
# Find Age, Cabin and Fare features have missing value for the test set.
myPrint(train.info(), printmessage="Training Data", hold=False)
myPrint(test.info(), printmessage="Test Data", hold=False)

### Look at the distributions of numerical features
myPrint(train.describe(), printmessage="Distribution of features", hold=False)

### Looks at distribution of categorical features (features that are in discrete groups) 
myPrint(train.describe(include=['O']), printmessage="Distribution of categorical features", hold=False)



#############################
##### - Data analysis - #####

# Here I analyse the different features in the data and determine which ones correlate with survival. 
# New features that are combinations of others which correlate with survival may be useful.
# Missing values should only be filled if they're useful and redundant data is dropped.
# All features need to be assigned numerical values in order to be used


### Look at how passenger class (Pclass) correlates with survival
myPrint(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False),
        printmessage="Correlation of class with survival",
        hold=False,)

### Look at how passenger sex correlates with survival
myPrint(train[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False),
        printmessage="Correlation of sex with survival",
        hold=False,)

### Look at how family size correlates with survival
myPrint(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False),
        printmessage="Correlation of family size with survival",
        hold=False,)

myPrint(train[['Ticket']].head(), hold=False) 
myPrint(train[['Cabin']].head(), hold=False)

## Split the ticket entry into the prefix (if any) and last number
for dataset in total:
    df_x = dataset['Ticket'].str.extract(r"(?P<TicketPrefix>\S+ )?(?P<TicketNumber>\d+)")
    dataset['TicketPrefix'] = df_x['TicketPrefix']
    dataset['TicketNumber'] = df_x['TicketNumber']
    ## Drop the Ticket option now no longer needed
    dataset.drop(["Ticket"], inplace=True, axis=1)

print(train.columns.values)
print(test.columns.values)

## Count different ticket numbers (should all be different) and ticket Prefixes
myPrint(train['TicketNumber'].value_counts(),hold=False)
myPrint(train['TicketPrefix'].value_counts(),hold=False)

## Look to see if any correlation between ticket number and ticket prefix and survival
## Ticket numbers all unique so remove
myPrint(train[['TicketNumber', 'Survived']].groupby(['TicketNumber'], as_index=False).mean().sort_values(by='Survived', ascending=False),
        printmessage="Correlation of TicketNumber with survival",
        hold=False,)

myPrint(train[['TicketPrefix', 'Survived']].groupby(['TicketPrefix'], as_index=False).mean().sort_values(by='Survived', ascending=False),
        printmessage="Correlation of TicketPrefix with survival",
        hold=False,)

## Split the cabin entries into the first letter and last number. 
for dataset in total:
    df_x = dataset["Cabin"].str.extract(r"(?P<CabinPrefix>[a-zA-Z])?(?P<CabinNumber>\d+)")
    dataset['CabinPrefix'] = df_x['CabinPrefix']
    dataset['CabinNumber'] = df_x['CabinNumber']
    ## Drop the Cabin option now no longer needed
    dataset.drop(["Cabin"], inplace=True, axis=1)

## Look at individual cabin numbers and prefixes
## Most cabin entries are NaN so unlikely to useful as a feature however.
myPrint(train['CabinNumber'].value_counts(),hold=False)
myPrint(train['CabinPrefix'].value_counts(),hold=False)

## Look to see if any correlation between cabin number and prefix and survival. 
myPrint(train[['CabinNumber', 'Survived']].groupby(['CabinNumber'], as_index=False).mean().sort_values(by='Survived', ascending=False),
        printmessage="Correlation of CabinNumber with survival",
        hold=False,)

myPrint(train[['CabinPrefix', 'Survived']].groupby(['CabinPrefix'], as_index=False).mean().sort_values(by='Survived', ascending=False),
        printmessage="Correlation of CabinPrefix with survival",
        hold=False,)




################################
### - Checking assumptions - ###

# Key assumptions -  If sex = female, more likely to survive. 
# If Pclass = 1, then most likely to survive, followed by 2,3. 
# People in large families or alone most likely died.

### First analyse age and survival for all passengers
# Can conclude that young children more likely to survive
histogramAge = sns.FacetGrid(train, col = 'Survived') # create instance of FacetGrid
histogramAge.map(plt.hist, 'Age', bins = 20) # map the grid to a plot in pyplot
histogramAge.fig.subplots_adjust(top=0.8)
histogramAge.fig.suptitle("Age vs survival for all passengers")
myPlot()

### Analyse survival of passengers for each class
# Can see you're more likely to not have survived if Pclass = 3 and more likely to survive if Pclass = 1
histogramPclassAge = sns.FacetGrid(train, col = 'Survived', row = 'Pclass', height = 2.2, aspect = 1.6)
histogramPclassAge.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
histogramPclassAge.add_legend()
histogramPclassAge.fig.subplots_adjust(top=0.9)
histogramPclassAge.fig.suptitle("Age vs survival for each class")
myPlot()

### Analyse passenger sex and age
# Can conclude that you were much less likely to survive as a young male 
# and more likely to survive as a woman 
histogramPclassAge = sns.FacetGrid(train, col = 'Survived', row = 'Sex', height = 2.2, aspect = 1.6)
histogramPclassAge.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
histogramPclassAge.add_legend()
histogramPclassAge.fig.subplots_adjust(top=0.9)
histogramPclassAge.fig.suptitle("Age vs survival for each sex")
myPlot()

### Analyse Embarking point and survival
## Instead of a bar chart, look at a pointplot which shows the mean value with an error bar. 
## Find that females more likely to survive if boarding at S and Q ports, males at C port.
## For Pclass = 3, survival of male and females closer than other classes
## Note that in the plot, have some data with only a few points, e.g, only two females (both survived)
## and one male (died) boarded at Q from Pclass = 2, and one male (died) and female (survived) from Pclass = 1.

lineplotEmbarkedPclass = sns.FacetGrid(train, row = 'Embarked', height = 2.2, aspect = 1.6)
lineplotEmbarkedPclass.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep', split=True, order=[1,2,3], hue_order=['female','male'])
lineplotEmbarkedPclass.add_legend()
lineplotEmbarkedPclass.fig.subplots_adjust(top=0.9)
lineplotEmbarkedPclass.fig.suptitle("Embarking point and survival")
myPlot()

myPrint(train[(train['Embarked'] == 'C') & train['Sex'] == 'Male'].head(), hold=False)


### Analyse Fare vs Embarkment and Fare vs Sex
# See those who embarked at Q paid a lower fare and had equal liklihood of suriviving and not surviving
# Those who embarked at C and S were more likely to survive if their fare was higher
histogramFareEmbarkSex = sns.FacetGrid(train, row = 'Embarked', col = 'Survived', height = 2.2, aspect = 1.6)
histogramFareEmbarkSex.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci = None, order=['female','male'])
histogramFareEmbarkSex.add_legend()
histogramFareEmbarkSex.fig.subplots_adjust(top=0.9)
histogramFareEmbarkSex.fig.suptitle("Fare vs Embarkment and Fare vs Sex")
myPlot()




###############################
### - Feature engineering - ###

### First we drop the ticket and cabin features as these aren't useful
## The cabin features aren't useful because there are so many missing entries
## The ticket feature we found didn't correlate with survival.
newtotal = list(map(lambda x: x.drop(['TicketPrefix', 'TicketNumber','CabinPrefix', 'CabinNumber'], axis = 1), total))


### Create 'Title' feature from the names and titles
## First extract the titles
for dataset in newtotal:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
myPrint(pd.crosstab(newtotal[0]['Title'], newtotal[0]['Sex']), hold=False)

## Take rare title and combine into one category. Also replace Ms with Miss and Mme with Mrs
for dataset in newtotal:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

## Calculate the mean survival rate for each title
myPrint(newtotal[0][['Title', 'Survived']].groupby(['Title'], as_index = False).mean(),
        printmessage="Mean Survival", hold=False)

## Now map the titles to numbers
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
for dataset in newtotal:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # fill any missing entries with 0
    dataset['Title'] = dataset['Title'].astype(int) # set type as int
myPrint(newtotal[0].head(), hold=False)

## Drop the name feature now the title feature created
newtotal2 = list(map(lambda x: x.drop(['Name', 'PassengerId'], axis = 1), newtotal))


### Creating Sex feature. 
sex_mapping = {'female': 1, 'male': 0}
for dataset in newtotal2:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)
myPrint(newtotal2[0].head(), hold=False)


### Creating Age feature

## Complete missing Age data -- First look at age data
ageGrid = sns.FacetGrid(newtotal2[0], row = 'Pclass', col = 'Sex', height = 2.2, aspect = 1.6)
ageGrid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
myPlot()

## Now interpolate ages for missing values. This is done either
## by replacing the age by the median for a given sex or by sampling from
## a uniform distribution using the mean and std of the sex and Pclass
for dataset in newtotal2:
    for i in range(0,2): # only two sex values - 0 and 1
        for j in range(0,3): # only 3 class values - 1, 2, 3 (add a 1 to value later

            # Obtain the data with the corresponding sex and class. Remove missing values to calculate the mean and standard deviation (std).
            guess = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna() 
           
            # Fill in any missing values using an age guess from a random distribution with a mean and std from the data
            # Note that rather than guessing from a random distribution using the mean and std from all ages,
            # we instead use the mean and std from each Sex and Pclass
            age_guess = rnd.uniform((guess.mean()-guess.std()),(guess.mean() + guess.std()))

            # Round age to nearest half a year
            guess_val = int(age_guess/0.5 + 0.5)*0.5
            print(i, " ", j, " ", guess_val)

            # Replace age with guess value if null entry
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_val
    dataset['Age'] = dataset['Age'].astype(int) # Convert ages to integer values

## Split data into age bands and check survival based on bands
train_ageband = newtotal2[0].copy()
train_ageband['AgeBand'] = pd.cut(train_ageband['Age'], 5)
myPrint(train_ageband[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values(by = 'AgeBand', ascending = True),
        hold=False)

## Split the total dataset into bands and then assign numerical values to each band
for dataset in newtotal2:
    dataset.loc[(dataset['Age'] <= 16), 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64), 'Age'] = 4

newtotal3 = newtotal2


### Create new Alone Feature

## Create new FamilySize feature by adding siblings and parents together to person value (hence +1)
## Then create Alone feature if FamilySize = 1
for dataset in newtotal3:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['Alone'] = 0 # first create the alone feature with a zero value
    dataset.loc[dataset['FamilySize'] == 1, 'Alone'] = 1 # then assign to it a value of 1 if a person is alone, 0 if not

myPrint(newtotal3[0][['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values(by = 'Survived', ascending = False),
         hold=False)
myPrint(newtotal3[0][['Alone', 'Survived']].groupby(['Alone'], as_index = False).mean(),  
        hold=False)

## Remove the 'FamilySize', 'Parch' and 'SibSp' features
newtotal4 = list(map(lambda x: x.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1), newtotal3))
myPrint(newtotal4[0].head(), hold=False)


### Complete the Embark feature.
## Find the most common value of the embark feature and use this to fill missing values
freq_port = newtotal4[0].Embarked.dropna().mode()[0]
for dataset in newtotal4:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
myPrint(newtotal4[0][['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by = 'Survived', ascending = False),
         hold=False)

## Map the embark feature to a numerical value
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
for dataset in newtotal4:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
myPrint(newtotal4[0].head(), hold=False)


### Complete the Fare feature
## Fill missing values with the median Fare value
for dataset in newtotal4:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].dropna().median())
myPrint(newtotal4[0].head(), hold=False)

## Create Fare band and convert to numerical values - first look at bands and survival rates
train_testband = newtotal4[0].copy()
train_testband['Fareband'] = pd.qcut(train_testband['Fare'], 4)
myPrint(train_testband[['Fareband', 'Survived']].groupby(['Fareband'], as_index = False).mean().sort_values(by = 'Fareband', ascending = True),
         hold=False)

## Create the fare bands and map to numerical values
for dataset in newtotal4:
    dataset.loc[(dataset['Fare'] <= 7.91), 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31), 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

newtotal5 = newtotal4




################################
##### - Train the model - #####

## Remove the labels from the train dataset
X_train = newtotal5[0].drop('Survived', axis = 1)
Y_train = newtotal5[0]['Survived']
X_test = newtotal5[1]

### Build the model
modelType = 'RandomForest'

if modelType == 'RandomForest':
    ## Use Random Forest for training model
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2)
elif modelType == 'XGBoost':
    ## Use XGBoost for training model
    model = xgb.XGBRegressor(objective="binary:logistic", random_state=42)
else:
    myPrint("Model not chosen")
    sys.exit()

## Fit the model
model.fit(X_train, Y_train)

## Predicted output
Y_pred = model.predict(X_test)
predictions = [round(value) for value in Y_pred]

## Assesment of model
Y_pred_train = model.predict(X_train)
predictions_train = [round(value) for value in Y_pred_train]
score = accuracy_score(Y_train, predictions_train)
#score = model.score(X_train, Y_train)
accuracy_model = round(score*100,2)
myPrint(accuracy_model, hold=False)



##### - Create a submission file for the Kaggle competition - #####
## Entries are passenger ID and if they survived (0/1 for deceased/survived)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission.csv', index=False)
  


