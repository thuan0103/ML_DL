import pandas as pd
import numpy as np

df = pd.read_csv("linear_regression_data.csv")
y = df['Y'].values
x = df['X'].values

w = np.random.randn()
b = np.random.randn() 
learning_rate = 0.01
epochs = 10000

def loss_funtion(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

for epoch in range(epochs):
    y_pred = w*x +b

    dw = -2*np.mean(x*(y-y_pred))
    db = -2*np.mean(y-y_pred)

    w -= learning_rate*dw
    b -= learning_rate*db

    if epoch % 10 == 0:
        loss = loss_funtion(y,y_pred)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

print(f"\nFinal model: y = {w:.4f} * X + {b:.4f}")