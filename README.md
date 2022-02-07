# Kaggle_Titanic
This is a repository for codes used for Kaggle competition: Titanic - Machine learning from disaster.
Codes are provided both as a Jupyter notebook and as standard python files.

This Kaggle competition involved determining whether or not a given person on the titanic would survive the disaster, given the provided data which included the passengers name, age, sex, cabin number, ticket number, boardning point, class and if they had any parents or siblings on board.

It involved analysing and tidy data, as many entries were missing, and key features needed to be determined to train the model. By looking at correlations between different features and survival, age, sex, class, whether the person was alone and the title ended up being the most appropriate features to train the model.
The model was trained on both a random forest classifier and using XGboost. It achieved the highest accuracy of 78% on the competition test set with XGboost.

