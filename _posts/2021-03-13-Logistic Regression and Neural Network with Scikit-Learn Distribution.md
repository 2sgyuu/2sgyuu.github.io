# IoT·인공지능·빅데이터 개론 및 실습 (2021년도, 1학기, M2177.004900_001)

## 3/12 Logistic Regerssion & Neural Network with Scikit-Learn

Adapted by Seonwoo Min from the "An Introduction to Machine Learning with Scikit-learn" tutorial (http://scikit-learn.org/stable/tutorial/basic/tutorial.html).

In this excercise, we will cover:

* Loading an example dataset & preprocessing
* Logistic regression & neural network models in scikit-learn
* Model training & prediction & evaluation
* Model save & load
* Homework

## 1. Loading an example dataset & preprocessing


```python
from sklearn.datasets import load_digits
data = load_digits()
print(data.keys())
```

    dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])
    


```python
# Data shape & statistics
print("Data: ", data['data'].shape)
print("Label:", data['target'].shape)

# Print the number of samples for each class
import numpy as np
#################### To Do #################################
for c in range(10):
    print("Class:", c, " Number:", np.sum(data['target'] == c))
############################################################
```

    Data:  (1797, 64)
    Label: (1797,)
    Class: 0  Number: 178
    Class: 1  Number: 182
    Class: 2  Number: 177
    Class: 3  Number: 183
    Class: 4  Number: 181
    Class: 5  Number: 182
    Class: 6  Number: 181
    Class: 7  Number: 179
    Class: 8  Number: 174
    Class: 9  Number: 180
    


```python
#############################################################
# Data Visaulization
#############################################################
import matplotlib.pyplot as plt
%matplotlib inline

#################### To Do #################################
# Hint: plt.imshow(data['data'][i].reshape(8,8), cmap=plt.cm.gray_r)
for c in range(10):
    i = 0
    while(1):
        if data['target'][i] == c:
            plt.subplot(2, 5, c+1)
            plt.axis('off')
            plt.imshow(data['data'][i].reshape(8,8), cmap=plt.cm.gray_r)
            plt.title('Class %d:' % c)
            break
        i += 1
############################################################
```


    
![png](output_5_0.png)
    



```python
#############################################################
# 1st Preprocessing
# Use the first 20 samples in each clss as test data
# Use the others as training data
#############################################################

#################### To Do #################################
test_indexes, train_indexes = [], []
num = [0] * 10
for i in range(len(data['target'])):
    if num[data['target'][i]] < 10: test_indexes.append(i)
    else : train_indexes.append(i)
    num[data['target'][i]] += 1

test_data, test_target = data['data'][test_indexes], data['target'][test_indexes]
train_data, train_target = data['data'][train_indexes], data['target'][train_indexes]
############################################################

print(test_data.shape)
print(train_data.shape)
```

    (100, 64)
    (1697, 64)
    


```python
#############################################################
# 2nd Preprocessing
# Let's use only 2 and 3 for binary classification
#############################################################

#################### To Do #################################
test_data23, test_target23 = test_data[(test_target == 2) ^ (test_target == 3)], test_target[(test_target == 2) ^ (test_target == 3)]
train_data23, train_target23 = train_data[(train_target == 2) ^ (train_target == 3)], train_target[(train_target == 2) ^ (train_target == 3)]
############################################################

print(test_data23.shape)
print(train_data23.shape)
```

    (20, 64)
    (340, 64)
    

## 2. Logistic regression & neural network models in scikit-learn

For full documentations refer to the following links: <br>
Logistic Regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html <br>
Neural network: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier


```python
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

LR = LogisticRegression(max_iter=1000, solver='sag')
NN = MLPClassifier(hidden_layer_sizes=(10), activation='relu', learning_rate_init=0.01, max_iter=1000)
```

## 3. Model training & prediction & evaluation


```python
#############################################################
# Logistic regression model
#############################################################
# Training
LR = LogisticRegression(max_iter=1, solver='sag')
LR.fit(train_data23, train_target23)

# Prediction
train_predict23 = LR.predict(train_data23)
test_predict23 = LR.predict(test_data23)
print("test_target     :", test_target23)
print("test_prediction :", test_predict23)

#################### To Do #################################
# Evaluation
train_acc23 = np.sum(train_target23 == train_predict23) / len(train_target23)
test_acc23 = np.sum(test_target23 == test_predict23) / len(test_target23)
############################################################

print("train_acc :", train_acc23)
print("test_acc  :", test_acc23)
```

    test_target     : [2 3 2 3 2 3 3 2 2 2 2 3 3 3 3 2 2 3 2 3]
    test_prediction : [2 3 2 3 2 3 3 2 2 2 2 3 3 3 3 2 2 3 2 3]
    train_acc : 0.9882352941176471
    test_acc  : 1.0
    

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_sag.py:330: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      "the coef_ did not converge", ConvergenceWarning)
    


```python
#############################################################
# Neural network model
#############################################################

#################### To Do #################################
# Training
NN = MLPClassifier(hidden_layer_sizes=(10), activation='relu', learning_rate_init=0.01, max_iter=1)
NN.fit(train_data23, train_target23)

# Prediction
train_predict23 = NN.predict(train_data23)
test_predict23 = NN.predict(test_data23)
print("test_target     :", test_target23)
print("test_prediction :", test_predict23)

# Evaluation
train_acc23 = np.sum(train_target23 == train_predict23) / len(train_target23)
test_acc23= np.sum(test_target23 == test_predict23) / len(test_target23)
print("train_acc :", train_acc23)
print("test_acc  :", test_acc23)
############################################################
```

    test_target     : [2 3 2 3 2 3 3 2 2 2 2 3 3 3 3 2 2 3 2 3]
    test_prediction : [3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3 3 3 2 3]
    train_acc : 0.7323529411764705
    test_acc  : 0.65
    

    /usr/local/lib/python3.7/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:571: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    

## 4. Model save & load


```python
# from sklearn.externals import joblib
import joblib
import os

if not os.path.exists('models'):
    os.makedirs('models')

# save
joblib.dump(NN, 'models/NN23.joblib') 

# load
NN_load = joblib.load('models/NN23.joblib') 

#################### To Do #################################
# Prediction
train_predict23 = NN_load.predict(train_data23)
test_predict23 = NN_load.predict(test_data23)
print("test_target     :", test_target23)
print("test_prediction :", test_predict23)

# Evaluation
train_acc23 = np.sum(train_target23 == train_predict23) / len(train_target23)
test_acc23= np.sum(test_target23 == test_predict23) / len(test_target23)
print("train_acc :", train_acc23)
print("test_acc  :", test_acc23)
############################################################
```

    test_target     : [2 3 2 3 2 3 3 2 2 2 2 3 3 3 3 2 2 3 2 3]
    test_prediction : [3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3 3 3 2 3]
    train_acc : 0.7323529411764705
    test_acc  : 0.65
    

## 5. Homework
Now it's your job to experiment with models and achieve higher accuracy on the  **<font color=red>on the entire dataset</font>**. <br>
Try different hyperparameter configurations and save the final model as "final_model.joblib" <br>
Submit the current **notebook file and the saved final model** on ETL.
* Maximum 10 points for >= 97% accuracy on the test set
* Maximum 8 points for >= 96% accuracy on the test set
* Maximum 6 points for >= 95% accuracy on the test set
* Maximum 4 points for >= 94% accuracy on the test set


```python
#############################################################
# Try different hyperparameters
# Final model training
#############################################################

#################### To Do #################################

############################################################
```


```python
#############################################################
# Final model test
# Load the final model and obatin the test accuracy
#############################################################

#################### To Do #################################

############################################################
```

### Describe what you did here
In this cell you should also write an explanation of what you did, any additional features that you implemented, and any visualizations or graphs that you make in the process of training and evaluating your model.
* Maximum 10 points

_Tell us here_
