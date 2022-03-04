# https://machinelearningmastery.com/dynamic-ensemble-selection-in-python/

#Ensemble model test


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Dropout, MaxPooling1D
from tensorflow.keras.layers import concatenate, MaxPooling1D

from tensorflow.keras import Input, Model
from keras.layers.merge import concatenate
from tensorflow.keras import optimizers
import os
import numpy
import data_reduction
########################################################################
# evaluate KNORA-U dynamic ensemble selection with a custom pool of algorithms
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deslib.des.knora_u import KNORAU
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split the dataset into train and test sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

########################################################################

########################################################################


fi_input = Input(shape=(X.shape[1], ), name='fi_input')
dense2 = Dense(32, activation="relu")(fi_input)
dropout2 = Dropout(0.2)(dense2)
output2 = Dense(32, activation="sigmoid")(dropout2)

# concat_model = concatenate([output, output2])
dense3 = Dense(64, activation="relu")(output2)
dropout3 = Dropout(0.2)(dense3)
output3 = Dense(1, activation="sigmoid")(dropout3)

dl_model = Model(inputs=fi_input, outputs=output3)

dl_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
dl_model.summary()



# define classifiers to use in the pool
classifiers = [
	LogisticRegression(),
	DecisionTreeClassifier(),
	GaussianNB(),
    dl_model]
# fit each classifier on the training set
for c in classifiers:
	c.fit(X_train, y_train)
# define the KNORA-U model
model = KNORAU(pool_classifiers=classifiers)
# fit the model
model.fit(X_train, y_train)
# make predictions on the test set
yhat = model.predict(X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (score))
# evaluate contributing models
for c in classifiers:
	yhat = c.predict(X_test)
	score = accuracy_score(y_test, yhat)
	print('>%s: %.3f' % (c.__class__.__name__, score))



# # evaluate KNORA-U with a random forest ensemble as the classifier pool
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from deslib.des.knora_u import KNORAU
# from sklearn.ensemble import RandomForestClassifier
# X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# # split the dataset into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# # define classifiers to use in the pool
# pool = RandomForestClassifier(n_estimators=1000)
# # fit the classifiers on the training set
# pool.fit(X_train, y_train)
# # define the KNORA-U model
# model = KNORAU(pool_classifiers=pool)
# # fit the model
# model.fit(X_train, y_train)
# # make predictions on the test set
# yhat = model.predict(X_test)
# # evaluate predictions
# score = accuracy_score(y_test, yhat)
# print('Accuracy: %.3f' % (score))
# # evaluate the standalone model
# yhat = pool.predict(X_test)
# score = accuracy_score(y_test, yhat)
# print('>%s: %.3f' % (pool.__class__.__name__, score))