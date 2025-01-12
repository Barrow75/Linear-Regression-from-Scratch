# Day 5: Prompt
# Creating Linear Regression Model from scratch

import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
print(torch.__version__)
print(np.__version__)
print(pd.__version__)
print(matplotlib.__version__)


# 1. Create a dataset
#   - Random numbers
sample_data = {
    "Time Studying (hours)": [1, 2, 4, 5, 6, 7, 9, 13, 14, 15],
    "Exam Grades (%)": [23, 30, 54, 62, 76, 81, 87,95,99, 100]
}
df = pd.DataFrame(sample_data)

df.to_csv("study_score.csv", index=False)

data_load = pd.read_csv("study_score.csv")
print(data_load)


#x = sample_data['Time Studying (hours)']
#y = sample_data['Exam Grades (%)']




# 1. Load data into an array
x_data = data_load["Time Studying (hours)"].to_numpy()
y_data = data_load["Exam Grades (%)"].to_numpy()

# 2. Calculate the means:
x_mean = np.sum(x_data)/len(x_data)
y_mean = np.sum(y_data)/len(y_data)

# 3. Covariance
covar = np.sum((x_data - x_mean) * (y_data - y_mean)) / (len(x_data) - 1)

# 4. Variance
variance = np.sum((x_data - x_mean)**2) / (len(x_data) - 1)

# 5. Slope (m) aka weights
m = covar / variance

# Intercept (b) aka bias
# y = mx + b => b = y - mx
b = y_mean - m*x_mean

# All together
y_predicted = m* x_data + b

# Visualize regression line
x_range = np.linspace(min(x_data), max(x_data), 100)
y_range = m * x_range + b
plt.plot(x_range, y_range, color='red', label=f'Regression Line: y = {m:.2f}x + {b:.2f}')

# Need to evalaute the model accuracy (MSE)
def MeanSquareError(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse
mse = MeanSquareError(y_data, y_predicted)

# Gradient Descent: iteratively adjusts the slope (m) and intercept (b) to minimize MSE
def grad_descent(x, y_true, m, b, learning_rate, iterations):
    n = len(y_true)
    for _ in range(iterations):
        y_pred = m * x + b
        dm = -(2/n) * np.sum((x*(y_true - y_pred)))
        db = -(2/n) * np.sum(y_true - y_pred)

        m -= learning_rate * dm
        b -= learning_rate * db

    return  m, b

learning_rate = 0.01
iterations = 100
m, b = grad_descent(x_data, y_data, 0, 0, learning_rate, iterations)

# prediction of hours studied relating to test scores
def predictions(hours, m , b):
    return m*hours + b
new_hours_studied = np.array([3, 8, 10, 12])
predicted_scores = predictions(new_hours_studied, m, b)

print(f"Mean Squared Error (MSE): {mse: .2f}")
print(f"Gradient Descent: slope:{m: .2f} and intercept {b: .2f}")
plt.scatter(x_data, y_data, color='blue', label='Data Points')
x_range = np.linspace(min(x_data), max(x_data), 100)

plt.scatter(new_hours_studied, predicted_scores, color='green', label='New Predictions', marker='x', s=100)

plt.title("Time Studied vs Exam Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
plt.show()