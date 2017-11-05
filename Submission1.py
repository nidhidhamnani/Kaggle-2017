# KAGGLE SUBMISSION 
# Roll Number: CS15BTECH11028
# Algorithm used: Lightgbm by Microsoft
# Best ROC AUC Score: 0.81081

import numpy as np
import csv
from csv import reader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb



def main():

	# Reading data from csv files
	# trainMatrix contains the training data 
	# testMatrix contains the test data 
	with open('Dataset/trainData.csv', 'rb') as f:
		    reader = csv.reader(f)
		    trainMatrix = list(reader)

	with open('Dataset/testData.csv', 'rb') as f:
	        reader = csv.reader(f)
	        testMatrix = list(reader)

	trainMatrix = trainMatrix[1:]
	testMatrix = testMatrix[1:]


	# Data Preprocessing
	# Steps followed:
	# 	1. Unknown is assigned 0.
	#	2. For strings data type, integer value is manually assigned.
	job = {'unknown':0,'admin.':1,'blue-collar':2,'entrepreneur':3,'housemaid':4, 'management':5, 'retired':6,
	 		'self-employed':7,'services':8,'student':9,'technician':10, 'unemployed':11}

	marital = {'unknown':0,'single':1, 'married':2, 'divorced':3}

	education = {'unknown':0,'basic.4y':1, 'basic.6y':2, 'basic.9y':3, 'high.school':4, 'illiterate':5,
				'professional.course':6,'university.degree':7}

	default = {'unknown':0, 'yes':1, 'no':2}

	housing = {'unknown':0, 'yes':1, 'no':2}

	loan = {'unknown':0, 'yes':1, 'no':2}

	contact = {'cellular':1, 'telephone':2}

	month = {'jan':1,'feb':2,'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 
			'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

	day_of_week = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}

	poutcome = {'failure':1, 'nonexistent': 2, 'success':3}


	X_trainSet = []
	Y_train = []
	X_testSet = []
	
	# Converting data from string to int/float and separating class labels from features
	for row in trainMatrix:
		l = [int(row[0]), job[row[1]], marital[row[2]], education[row[3]], default[row[4]], housing[row[5]], 
        	loan[row[6]], contact[row[7]], month[row[8]], day_of_week[row[9]], int(row[10]), int(row[11]),
        	int(row[12]), poutcome[row[13]],float(row[14]), float(row[15]), float(row[16]), float(row[17]), float(row[18])]
        	# To decrease bias, adding duplicate entries for class label 1
        	if int(row[19])==1:
        		Y_train.append(int(row[19]))
        		X_trainSet.append(l)
        	Y_train.append(int(row[19]))
        	X_trainSet.append(l)
        	

        for row in testMatrix:
		l = [int(row[1]), job[row[2]], marital[row[3]], education[row[4]], default[row[5]], housing[row[6]], 
        	loan[row[7]], contact[row[8]], month[row[9]], day_of_week[row[10]], int(row[11]), int(row[12]),
        	int(row[13]), poutcome[row[14]], float(row[15]), float(row[16]), float(row[17]), float(row[18]), float(row[19])]
        	X_testSet.append(l)
	        	

	# Converting list to np array
	X=np.array(X_trainSet)
	Y=np.array(Y_train)
	X_test=np.array(X_testSet)

	# Normalizing the data
	# xmax = maximum value for a particular feature
	# xmin = minimum value for a particular feature
	# Normalization techinique used: Min-Max Normalization [(value-min)/(max-min)]
	xmax = np.amax(X,axis=0)
    	xmin = np.amin(X,axis=0)
    	continuous = [0,10,11,12,14,15,16,17,18]
    	for i in range(len(X_trainSet)):
        	for j in continuous:
            		data = float(X[i][j]-xmin[j])/(xmax[j]-xmin[j])
            		X[i][j] = data

        for i in range(len(X_testSet)):
        	for j in continuous:
            		data = float(X_test[i][j]-xmin[j])/(xmax[j]-xmin[j])
            		X_test[i][j] = data
			
	# Setting parameters for lightgbm
	params = {
	    'boosting_type': 'gbdt',
	    'objective': 'binary',
	    'metric': 'binary_logloss',
	    'num_leaves': 31,
	    'learning_rate': 0.05,
	    'feature_fraction': 0.9,
	    'bagging_fraction': 0.8,
	    'bagging_freq': 5,
	    'verbose': 0
	}

	params['metric'] = ['auc', 'binary_logloss']

	# Cross Validation
	# skf = StratifiedKFold(n_splits=5)
	# skf.get_n_splits(X,Y)
	# for trainindex,testindex in skf.split(X, Y):
	# 	x_train, x_test = X[trainindex], X[testindex]
	# 	y_train, y_test = Y[trainindex], Y[testindex]
	# 	num_round=30
	# 	train_data=lgb.Dataset(x_train, y_train)
	# 	lgbm=lgb.train(params,train_data,num_round)
	# 	predicted=lgbm.predict(x_test)
	# 	accuracy_lgbm = roc_auc_score(y_test,predicted)
	# 	print accuracy_lgbm

	# Training lightgbm
	train_data=lgb.Dataset(X, Y_train)
	lgbm=lgb.train(params,train_data,30)
	# Storing predicted values in predicted
	predicted=lgbm.predict(X_test)

	j=0
	# Storing the predicted values in a file
        with open('./Submission1.csv', 'wb') as csvfile:
			file = csv.writer(csvfile,dialect='excel')
			file.writerow(["id","class"])
			for row in predicted:
				j=j+1
				file.writerow([j,row])
				
			
				
		

main()

# For proper indendation use tab size: 8
