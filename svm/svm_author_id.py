#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

import time
from sklearn import svm
clf=svm.SVC(kernel='rbf', C=10000);

print(clf.kernel)
t=time.time();

clf.fit(features_train, labels_train);
#clf.fit(features_train[:len(features_train)/100], labels_train[:len(labels_train)/100]);
print(time.time()-t);
t=time.time();

pred=clf.predict(features_test);
print(time.time()-t);
t=time.time();

a=0;
for i in range(0, len(features_test)-1):
    if pred[i]==1:
        a=a+1;

print(a)


from sklearn.metrics import accuracy_score
import numpy as np
accuracy = accuracy_score(pred, np.ones(len(features_test)));
print(accuracy*len(features_test));

#10k ->
#1k ->
#100 ->
#10 ->