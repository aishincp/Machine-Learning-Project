#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Without Classification.

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_excel ('P2 X = 200 and Y = 10000.xlsx') #change the file name only 
df.drop(df.columns[0], axis=1, inplace=True)
df['Target'] = 2
df.rename(columns = {'Class':'Predicted'}, inplace = True)


X = df.Predicted
y = df.Target

y_true = y
y_pred = X



#Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
print(cm)




import seaborn as sns
import matplotlib.pyplot as plt     
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix \n Without using FFT File\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Empty', 'Human'])
ax.yaxis.set_ticklabels(['Empty', 'Human'])
## Display the visualization of the Confusion Matrix.
plt.show()

print(classification_report(y_true, y_pred))

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

#Precision 

Precision = TP/(TP+FP)

#Recall 

Recall = TP/(TP+FN)

#F1 score

f1_score = (Precision*Recall)/(Precision+Recall)


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print('TPR: '+str(TPR))
print('TNR: '+str(TNR))
print('PPV: '+str(PPV))
print('NPV: '+str(NPV))
print('FPR: '+str(FPR*100))
print('FNR: '+str(FNR*100))
print('FDR: '+str(FDR))
print('ACC: '+str(ACC))


# In[37]:


#with Classification 

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import category_encoders as ce

# label the data and remove the unwanted column
df = pd.read_csv('fft_P2 Winter Wear noMovement X200Y10000_.txt', delimiter="\t") # only change file name 
df1 = df.drop(['V0,2', '2,48', '0,0'], axis = 1) # check the fft file first and as per that change it, because in this only int value to train data. 
df1.drop(df.columns[0], axis=1, inplace=True)
df1['Target'] = 2 # Creating a traget column with respect the usecase.
df1.rename(columns = {'1':'Predicted'}, inplace = True)#label the data.
X = df1.drop(['Predicted'],axis=1).values
y = df1.Predicted



# Split the data in train and test form. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=26)



# initiate an rf classifier using a pipeline
clf = make_pipeline(
    SimpleImputer(strategy="mean"), RandomForestClassifier(random_state=26))

# train the classifier on training data
clf.fit(X_train, y_train)

# make predictions on test data
pred = clf.predict(X_test)

#Generate the confusion matrix
cm1 = confusion_matrix(y_test, pred, labels=[1, 2])
print(cm1)

import seaborn as sns
import matplotlib.pyplot as plt     
import seaborn as sns
import matplotlib.pyplot as plt     
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
cm1.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
cm1.flatten()/np.sum(cm1)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cm1, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix  \n with using FFT File');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Empty', 'Human'])
ax.yaxis.set_ticklabels(['Empty', 'Human'])
## Display the visualization of the Confusion Matrix.
plt.show()

print(classification_report(y_test, pred))


FP = cm1.sum(axis=0) - np.diag(cm1)
FN = cm1.sum(axis=1) - np.diag(cm1)
TP = np.diag(cm1)
TN = cm1.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print('TPR: '+str(TPR))
print('TNR: '+str(TNR))
print('PPV: '+str(PPV))
print('NPV: '+str(NPV))
print('FPR: '+str(FPR*100))
print('FNR: '+str(FNR*100))
print('FDR: '+str(FDR))
print('ACC: '+str(ACC))


# In[20]:


# split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[39]:


# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier


# instantiate the classifier 

rfc = RandomForestClassifier(random_state=0)


# fit the model

rfc.fit(X_train, y_train)


# Predict the Test set results

y_pred = rfc.predict(X_test)


# Check accuracy score 

from sklearn.metrics import accuracy_score

print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[40]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)


# In[42]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[27]:


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)


# In[32]:


# instantiate the classifier with n_estimators = 100

rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)



# fit the model to the training set

rfc_100.fit(X_train, y_train)



# Predict on the test set results

y_pred_100 = rfc_100.predict(X_test)



# Check accuracy score 

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))


# In[29]:


# create the classifier with n_estimators = 100

clf = RandomForestClassifier(n_estimators=100, random_state=0)



# fit the model to the training set

clf.fit(X_train, y_train)


# In[14]:


# create the classifier with n_estimators = 100

clf = RandomForestClassifier(n_estimators=100, random_state=0)



# fit the model to the training set

clf.fit(X_train, y_train)

