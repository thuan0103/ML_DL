import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data_classification.csv', header=None)
true_x = [item[0] for item in df.values if item[2] == 1]
false_x = [item[0] for item in df.values if item[2] == 0]

true_y = [item[1] for item in df.values if item[2] == 1]
false_y = [item[1] for item in df.values if item[2] ==0]

def sig_moid(z):
    return 1/(1+np.exp(-z))

def division(p):
    if p >= 0.5:
        return 1
    else:
        return 0
    
def predict(features, weight):
    z = np.dot(features,weight)
    return sig_moid(z)

def cost_function(features, labels, weight):
    # param features: 100x3
    # param labels: 100x1
    # param weights: 3x1
    n = len(labels)
    predictions = predict(features,weight)
    cost_class1 = -labels*np.log(predictions)
    cost_class2 = -(1-labels)*np.log(1-predictions)
    cost = cost_class1 + cost_class2
    return cost.sum()/n

def update_weight(features, labels, weight, learning_rate):
    n = len(labels)
    predictions = predict(features, weight)
    gradient = np.dot(features.T,(predictions - labels))
    weight -= learning_rate * gradient
    return weight

def train(features, labels, weight, learning_rate, epochs):
    costs = []
    for i in range(epochs):
        cost = cost_function(features, labels, weight)
        costs.append(cost)
        weight = update_weight(features, labels, weight, learning_rate)
        if i % 100 == 0:  # In ra cost mỗi 100 epoch
            print(f"Epoch {i}, Cost: {cost}")
    return weight, costs

features = np.hstack((np.ones((df.shape[0], 1)), df[[0, 1]].values))
labels = df[2].values.reshape(-1,1)
weight = np.zeros((features.shape[1], 1))
learning_rate = 0.01
epochs = 5000
final_weight, costs = train(features, labels, weight, learning_rate, epochs)

plt.plot(costs)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost Function over Epochs")
plt.show()

x_values = [min(features[:, 1]), max(features[:, 1])]
print(x_values)
y_values = -(final_weight[0] + final_weight[1] * np.array(x_values)) / final_weight[2]

plt.scatter(true_x, true_y, marker='o', c='b', label='Class 1')
plt.scatter(false_x, false_y, marker='o', c='r', label='Class 0')
plt.plot(x_values, y_values, label="Decision Boundary")
plt.legend()
plt.show()