# Binary Classification and Optimization Project
This project implements binary classification with Regularization and Optimization
it contains two features :
- Test 1 , Test 2 ( X1 , X2 independent variables )
- one target ( Accepted ) represents in ( 0 , 1 ) ( y dependent variable )

## Features
- visualizes accepted and rejected students
- Feature Mapping
- implements sigmoid function , - cost , gradient manully
- adds lambda regularization to reduce overfitting
- implements optimization and regularization
- converts probabilities into binary predictions by this following code ```
    probability = sigmoid( X * theta.T )
    return [ 1 if x >= 0.50 else 0 for x in probability ] ```

- calculates  Model's Accurcy


## Data Structure
- Test 1 : the score in the first exam
- Test 2 : the score in the second exam
- Accepted : Admission result ( 1 = Accepted , 0 = Rejected )


## Implementation Details

- clear the terminal
- load the data and displays its information
- splits Accepted Student and Rejected Student
- shows a figure of Accepted Student and Rejected Student
- inserts the column of the ones

- creates polynomial features up to degree 6 to enable logistic regression
  to learn a non-linear decision boundary
  ``` python degree = 6 

for i in range(1 , degree + 1   ):
    for j in range( 0 , i):
        data[f"F{i}{j}"] =  np.power(x1 , i - j ) * np.power( x2 , j ) ```

- splits X features and y target
- initializes theta
- implements sigmoid function
  mathematical equation of sigmoid = 1 / ( 1 + e^ (- X 0) )
- calculates mathematically cost function
- implements gradient and adds a lambda ( penalty ) to reduce overfitting and theta values small
- optimizes the model parameters using Scipy optimization to minimize the cost function
- sets the target to zero or one
- predicting new predictions
- calculates Model's Accuracy


## Result

```
result = (array([  0.68975378,  -1.85084734,   9.31329816,  -6.7344271 ,
        12.23343196,  24.86665454,  10.07403536, -15.76787273,
        -5.13606792, -21.65344469,   0.23050047,  -2.87157687,
       -15.04779415,  17.41208679, -14.42661723, -23.99118004,
       -18.69712165,  -6.04226655,  -3.13282086,  -4.35114792,
       -32.42243902,  16.26389868]), 108, 1)

cost before optimization = 0.6931471805599454
cost after optimization = 0.409355569790105

new predictions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]

Accuracy = 79.66101694915254 %
```


## Visualization
- the figure below shows Accepted and Rejected Students
![Admitted and Not Admitted Students](images/figure.png)


## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Hassan-mahmoud33/regularization.git
```

2. Navigate to the project directory:

```bash
cd regularization
```

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib scipy
```

4. Run the project:

```bash
python regularization.py
```

## Libraries used 
- Numpy
- matplotlib
- pandas
- os
- scipy

  ## Purpose
  - this project for educational purposes









