import numpy as np
import random

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.weights = None

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, weights):
        num_examples = y.size
        h = self.sigmoid(np.dot(X, weights))
        cost = (-1 / num_examples) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)

        self.weights = np.zeros(X.shape[1])

        for i in range(self.num_iterations):
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)

        return np.round(self.sigmoid(np.dot(X, self.weights)))


if __name__ == '__main__':
    # Example usage with random data
    np.random.seed(0)
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, size=100)

    # Create model
    model = LogisticRegression(learning_rate=0.1, num_iterations=3000)

    # Fit the model
    model.fit(X, y)

    # Predict on new data
    X_test = np.random.rand(10, 3)
    y_pred = model.predict(X_test)

    print(y_pred)
