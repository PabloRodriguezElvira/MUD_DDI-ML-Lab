#!/usr/bin/env python3

import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import numpy as np
from joblib import dump


def load_data(data):
	features = []
	labels = []
	for interaction in data:
		interaction = interaction.strip()
		interaction = interaction.split('\t')
		interaction_dict = {feat: 1 for feat in interaction[1:] }
		features.append(interaction_dict)
		labels.append(interaction[0])
	return features, labels


if __name__ == '__main__':

	model_file = sys.argv[1]
	vectorizer_file = sys.argv[2] 	

	train_features, y_train = load_data(sys.stdin)
	y_train = np.asarray(y_train)
	v = DictVectorizer()
	X_train = v.fit_transform(train_features)

	clf = LinearSVC(C=0.1, class_weight='balanced', max_iter=2000)
	clf.fit(X_train, y_train)

	#Save classifier and DictVectorizer
	dump(clf, model_file) 
	dump(v, vectorizer_file)

	 






		


	 
