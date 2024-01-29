import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class LinearRegression():
    #initialize an object with learning rate and # of epochs
    def __init__(self, learningRate, epochs):
        self.learningRate = learningRate
        self.epochs = epochs

    def gradientDescent(self):
        #get the predicted Y
        self.Y_pred = self.predict(self.x)

        #calculate gradient for weight and bias
        dw = (1/self.m) * np.dot(self.x.T, (self.Y_pred-self.y))
        db = (1/self.m) * np.sum(self.Y_pred-self.y)

        #reassign weight and bias
        self.weight = self.weight - self.learningRate * dw
        self.bias = self.bias - (self.learningRate * db)

    def fit(self, x, y):
        #get rows and columns of training data
        self.m, self.n = x.shape

        #initialize and array of zeroes at the size of columns in training data
        self.weight = np.zeros(self.n)
        #initialize bias constant to 0
        self.bias = 0

        #assign the training data to object
        self.x = x
        self.y = y
        #print(self.y.shape)

        #use gradient descent to adjust the weights
        for x in range(self.epochs):
            self.gradientDescent()
        return self

    def predict(self, X):
        return np.dot(X, self.weight) + self.bias

    def h(self):
        return self.x@self.weight

    def mse(self, y_test, predictions):
        return np.mean((y_test - predictions) ** 2)


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
#separate data and independent variables
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
x = data
y = target

#split the data set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state=5)

#train the model
model = LinearRegression( 0.000002, 380)
model.fit(X_train,Y_train)

#run test through prediction
results = model.predict(X_test)
#calculate mean squared error results
errorResults = model.mse(Y_test, results)
print(errorResults)
