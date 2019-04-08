#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:41:17 2019

@author: siddharth
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:58:44 2019

@author: siddharth
"""

import pandas as pd
from sklearn.metrics import precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


#reading the file
df = pd.read_csv('zoo_data.csv')

#splitting into independent and dependent set
X = df.iloc[:,1:17].values
y = df.iloc[:,[17]].values



#binarizing for multiclass SVM
Y = label_binarize(y, classes=[1, 2,3,4,5,6,7])
n_classes = Y.shape[1]



#Train-Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=0)





# Run required classifier
classifier = OneVsRestClassifier(SVC(kernel = 'linear', random_state = 0))
#classifier = OneVsRestClassifier(SVC(kernel = 'poly', random_state = 0))
#classifier = OneVsRestClassifier(SVC(kernel = 'rbf', random_state = 0))
#classifier = OneVsRestClassifier(SVC(kernel = 'poly', probability =True ,random_state = 0))



#fitting the model
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)





#The average precision score in multi-label settings
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])




# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))









#plotting the required curve
plt.plot(recall["micro"], precision["micro"],
         label='micro-average Precision-recall curve(linear) (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i in range(n_classes):
    plt.plot(recall[i], precision[i],
             label='Precision-recall curve of class(Linear) {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class(poly)')
plt.legend(loc = 'centre left')
plt.show()

