# %% [markdown]
# # SWMAL Exercise
# 
# ## Implementing a dummy binary-classifier with fit-predict interface
# 
# We begin with the MNIST data-set and will reuse the data loader from Scikit-learn. Next we create a dummy classifier, and compare the results of the SGD and dummy classifiers using the MNIST data...
# 
# #### Qa  Load and display the MNIST data
# 
# There is a `sklearn.datasets.fetch_openml` dataloader interface in Scikit-learn. You can load MNIST data like 
# 
# ```python
# from sklearn.datasets import fetch_openml
# # Load data from https://www.openml.org/d/554
# X, y = fetch_openml('mnist_784',??) # needs to return X, y, replace '??' with suitable parameters! 
# # Convert to [0;1] via scaling (not always needed)
# #X = X / 255.
# ```
# 
# but you need to set parameters like `return_X_y` and `cache` if the default values are not suitable! 
# 
# Check out the documentation for the `fetch_openml` MNIST loader, try it out by loading a (X,y) MNIST data set, and plot a single digit via the `MNIST_PlotDigit` function here (input data is a 28x28 NMIST subimage)
# 
# ```python
# %matplotlib inline
# def MNIST_PlotDigit(data):
#     import matplotlib
#     import matplotlib.pyplot as plt
#     image = data.reshape(28, 28)
#     plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
#     plt.axis("off")
# ```
# 
# Finally, put the MNIST loader into a single function called `MNIST_GetDataSet()` so you can reuse it later.

# %%
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Function to plot an MNIST digit
def MNIST_PlotDigit(data):
    import matplotlib
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

# Function to load MNIST dataset
def MNIST_GetDataSet():
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, cache=True)
    X = X / 255.0  # Scale pixel values to [0, 1]
    return X, y

# # Load the dataset
# X, y = MNIST_GetDataSet()

# # Plot a single digit. The first one in the dataset
# MNIST_PlotDigit(X[2])

# print(X.shape)


# %% [markdown]
# #### Qb  Add a Stochastic Gradient Decent [SGD] Classifier
# 
# Create a train-test data-set for MNIST and then add the `SGDClassifier` as done in [HOML], p.103.
# 
# Split your data and run the fit-predict for the classifier using the MNIST data.(We will be looking at cross-validation instead of the simple fit-predict in a later exercise.)
# 
# Notice that you have to reshape the MNIST X-data to be able to use the classifier. It may be a 3D array, consisting of 70000 (28 x 28) images, or just a 2D array consisting of 70000 elements of size 784.
# 
# A simple `reshape()` could fix this on-the-fly:
# ```python
# X, y = MNIST_GetDataSet()
# 
# print(f"X.shape={X.shape}") # print X.shape= (70000, 28, 28)
# if X.ndim==3:
#     print("reshaping X..")
#     assert y.ndim==1
#     X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
# assert X.ndim==2
# print(f"X.shape={X.shape}") # X.shape= (70000, 784)
# ```
# 
# Remember to use the category-5 y inputs
# 
# ```python
# y_train_5 = (y_train == '5')    
# y_test_5  = (y_test == '5')
# ```
# instead of the `y`'s you are getting out of the dataloader. In effect, we have now created a binary-classifier, that enable us to classify a particular data sample, $\mathbf{x}(i)$ (that is a 28x28 image), as being a-class-5 or not-a-class-5. 
# 
# Test your model on using the test data, and try to plot numbers that have been categorized correctly. Then also find and plots some misclassified numbers.

# %%
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

# load the data
# X, y = MNIST_GetDataSet()

# reshape if needed
# print(f"X.shape={X.shape}") # print X.shape= (70000, 28, 28)
# if X.ndim==3:
#     print("reshaping X..")
#     assert y.ndim==1
#     X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
# assert X.ndim==2
# print(f"X.shape={X.shape}") # X.shape= (70000, 784)

# # split data set
# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# # What are we looking fore
# y_train_5 = (y_train == '5') # True for all 5s, False for all other digits
# y_test_5 = (y_test == '5')

# # Train
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)

# # Test
# y_pred = sgd_clf.predict(X_test)

# Score
# #identify the correct answers
# print("the ones where we have quessed true")
# for i in range(10):
#     if y_pred[i] == False and y_test_5[i] == False:
#         print(f"y_pred: {y_pred[i]} y_test {y_test[i]}")
#         print(f"True Negative{i}")

# for i in range(50):
#     if y_pred[i] == True and y_test_5[i] == True:
#         print(f"y_pred: {y_pred[i]} y_test {y_test[i]}")
#         print(f"True Positive{i}")

# MNIST_PlotDigit(X_test[7])
# MNIST_PlotDigit(X_test[15])




# %% [markdown]
# #### Qc Implement a dummy binary classifier
# 
# Now we will try to create a Scikit-learn compatible estimator implemented via a python class. Follow the code found on p.107 3rd [HOML] (for [HOML] 1. and 2. editions: name you estimator `DummyClassifier` instead of `Never5Classifyer`).
# 
# Here our Python class knowledge comes into play. The estimator class hierarchy looks like
# 
# <img src="https://itundervisning.ase.au.dk/SWMAL/L02/Figs/class_base_estimator.png" alt="WARNING: could not get image from server." style="width:500px">
# 
# All Scikit-learn classifiers inherit from `BaseEstimator` (and possibly also `ClassifierMixin`), and they must have a `fit-predict` function pair (strangely not in the base class!) and you can actually find the `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin` python source code somewhere in you anaconda install dir, if you should have the nerves to go to such interesting details.
# 
# But surprisingly you may just want to implement a class that contains the `fit-predict` functions, ___without inheriting___ from the `BaseEstimator`, things still work due to the pythonic 'duck-typing': you just need to have the class implement the needed interfaces, obviously `fit()` and `predict()` but also the more obscure `get_params()` etc....then the class 'looks like' a `BaseEstimator`...and if it looks like an estimator, it _is_ an estimator (aka. duck typing).
# 
# Templates in C++ also allow the language to use compile-time duck typing!
# 
# > https://en.wikipedia.org/wiki/Duck_typing
# 
# Call the fit-predict on a newly instantiated `DummyClassifier` object, and find a way to extract the accuracy `score` from the test data. You may implement an accuracy function yourself or just use the `sklearn.metrics.accuracy_score` function. 
# 
# Finally, compare the accuracy score from your `DummyClassifier` with the scores found in [HOML] "Measuring Accuracy Using Cross-Validation", p.107. Are they comparable? 
# 
# (Notice that Scikit-learn now also have a `sklearn.dummy.DummyClassifier`, but you are naturally supposed to create you own...)

# %%
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

import numpy as np

class DummyClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        pass

    def predict(self, X):
        assert X.ndim == 2
        return np.zeros(X.shape[0],dtype=bool)
    
