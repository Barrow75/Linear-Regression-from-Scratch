# Linear-Regression-from-Scratch
A deep look into Machine Learning topics such as, Gradient Descent, Mean Square Error (MSE), Linear Regression, etc. As well as the math that is behind it implemented with Python


Steps to complete this:
  1. Create a two variable data set of things that can be correlated together
       Example: How does the number of hours spent studying affect test scores? Hours Studied (x) vs Exam Scores (y)
```
     sample_data = {
    "Time Studying (hours)": [1, 2, 4, 5, 6, 7, 9, 13, 14, 15],
    "Exam Grades (%)": [23, 30, 54, 62, 76, 81, 87,95,99, 100]
}
```
  2. Save the data into a CSV file and then load it to be able to have the code read from the file
     ```
     df = pd.DataFrame(sample_data)

    df.to_csv("study_score.csv", index=False)

    data_load = pd.read_csv("study_score.csv")
    print(data_load)
    ```


3. Calculate the linear regression
**Linear Regression**
- Method of making accurate predictions by finding the best fit line or model for the data by finding the linear relationship between two variables
- Can be used for evaluating trends
- Types of Linear Regression
    + Simple Linear Regression *(One I used in my code)*
    + Multiple Linear Regression

    - Steps of calculating the Linear Regression:
        1. Load the data from the file 
        2. Calculate the mean of both x and y
        3. Calculate the Covariance and the Variance
        4. Find the slope m aka weights => Covariance / Variance
        5. Find the intercept aka bias => mean of y - m * mean of x
        6. The predicting y values (exam grades) for given x (hours studied)

4. Calculate the Mean Square Error (MSE)
   
   **Mean Squared Error**
   - Compute the error between tactual and predicted values
   - The lower the MSE the better the model fits the data
  
5. Calculate Gradient Descent
**Gradient Descent**
- Helps train models by iteratively adjusting their parameters in the direction of the steepest descent to minimize cost function
    - Steps to calculating Gradient Descent
      ```
        def grad_descent(x, y_true, m, b, learning_rate, iterations):
            n = len(y_true) # finds total number of data points (n) and used to compute average gradients
            for _ in range(iterations): # iterates to refine m and b
                y_pred = m * x + b # slope (computes the predicted y values using current slope and intercept; compare true y values to compute error
                dm = -(2/n) * np.sum(x * (y_true - y_pred))
                db = -(2/n) * np.sum(y_true - y_pred)
                m -= learning_rate * dm
                b -= learning_rate * db
            return m, b
      ```
  6. Make predictions with new data
      
