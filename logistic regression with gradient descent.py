import random
import math

def add_intercept(X):
    intercept = [[1] for _ in range(len(X))]
    return [intercept[i] + X[i] for i in range(len(X))]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def cost_function(X, y, weights):
    num_examples = len(y)
    cost = 0
    
    for i in range(num_examples):
        z = 0
        for j in range(len(X[0])):
            z += weights[j] * X[i][j]
        h = sigmoid(z)
        cost += -y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)
        
    cost /= num_examples
    
    return cost

def gradient_descent(X, y, weights, learning_rate, num_iterations):
    num_examples, num_features = len(X), len(X[0])
    
    for i in range(num_iterations):
        gradient = [0] * num_features
        for j in range(num_examples):
            z = 0
            for k in range(num_features):
                z += weights[k] * X[j][k]
            h = sigmoid(z)
            for k in range(num_features):
                gradient[k] += (h - y[j]) * X[j][k]
        
        for k in range(num_features):
            gradient[k] /= num_examples
            weights[k] -= learning_rate * gradient[k]
    
    return weights

def predict(X, weights):
    X = add_intercept(X)
    y_pred = []
    for i in range(len(X)):
        z = 0
        for j in range(len(X[0])):
            z += weights[j] * X[i][j]
        h = sigmoid(z)
        y_pred.append(round(h))
    
    return y_pred

if __name__ == '__main__':
    # Example usage with random data
    random.seed(0)
    X = [[random.random() for _ in range(3)] for _ in range(100)]
    y = [random.randint(0, 1) for _ in range(100)]
    
    # Add intercept
    X = add_intercept(X)

    # Initialize weights
    weights = [0] * len(X[0])

    # Train the model
    weights = gradient_descent(X, y, weights, learning_rate=0.1, num_iterations=3000)

    # Predict on new data
    y_pred = predict(X, weights)

    # Print results
    print('Weights:', weights)
    print('Accuracy:', sum([1 if y_pred[i] == y[i] else 0 for i in range(len(y))]) / len(y))
