import random
import math

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
    
    def add_intercept(self, X):
        intercept = [[1] for _ in range(len(X))]
        return [intercept[i] + X[i] for i in range(len(X))]
    
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
    
    def cost_function(self, X, y, weights):
        num_examples = len(y)
        cost = 0
        
        for i in range(num_examples):
            z = 0
            for j in range(len(X[0])):
                z += weights[j] * X[i][j]
            h = self.sigmoid(z)
            cost += -y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)
            
        cost /= num_examples
        
        return cost
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        num_examples, num_features = len(X), len(X[0])
        self.weights = [0] * num_features
        
        for i in range(self.num_iterations):
            gradient = [0] * num_features
            for j in range(num_examples):
                z = 0
                for k in range(num_features):
                    z += self.weights[k] * X[j][k]
                h = self.sigmoid(z)
                for k in range(num_features):
                    gradient[k] += (h - y[j]) * X[j][k]
            
            for k in range(num_features):
                gradient[k] /= num_examples
                self.weights[k] -= self.learning_rate * gradient[k]
    
    def predict(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        y_pred = []
        for i in range(len(X)):
            z = 0
            for j in range(len(X[0])):
                z += self.weights[j] * X[i][j]
            h = self.sigmoid(z)
            y_pred.append(round(h))
        
        return y_pred

if __name__ == '__main__':
    # Example usage with random data
    random.seed(0)
    X = [[random.random() for _ in range(3)] for _ in range(1000)]
    y = [random.randint(0, 1) for _ in range(1000)]

    # Create model
    model = LogisticRegression(learning_rate=0.1, num_iterations=3000)

    # Fit the model
    model.fit(X, y)

    # Predict on new data
    y_pred = model.predict(X)

    # Print results
    print('Weights:', model.weights)
    print('Accuracy:', sum([1 if y_pred[i] == y[i] else 0 for i in range(len(y))]) / len(y))
