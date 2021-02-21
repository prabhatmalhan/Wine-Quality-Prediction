'''
WINE QUALITY PREDICTION

Project By :-
PRABHAT MALHAN
B.Tech (Hons.)
graphic Era Deemed to be University
'''
# IMPORTING THE MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score


# READING DATASET
dataset = pd.read_csv("./Wine_Quality_Dataset.csv")


# CATAGORIZING THE QUALITY IN DATASET
print('Preparing data for Training...')
quality = list()
for i in dataset.iloc[:,-1]:
    if i<5: quality.append('bad')
    elif i<7: quality.append('good')
    else :quality.append('best')
dataset['quality']=quality
print('Data Ready For Training...!')


# TRAINING THE MODEL
featureX = np.asarray(dataset.iloc[:,:-1])
featureY = dataset['quality'].values
x_train, x_test, y_train, y_test = train_test_split(featureX, featureY, test_size=0.2, random_state=0)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.375, random_state=0)


# PRERARING THE MODEL FOR PREDICTION
print('\nTraining the model...')
classifier_model = RandomForestClassifier(n_estimators=200,random_state=459)
classifier_model.fit(x_train, y_train)
classifier_model.fit(x_validation, y_validation)
print('Model ready for dumping...!')


# DAMPING THE MODEL FOR FURTHER USE
import pickle
Model_Dumping_Path = open('model.pkl','wb')
print('\nDumping Started...')
pickle.dump(classifier_model,Model_Dumping_Path)
print('Model dumped...!')
Model_Dumping_Path.close()