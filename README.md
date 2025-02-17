
in this project, i generated datasets specifically designed to exploit the strengths and weaknesses of linear regression and decision tree learners. my goal was to create data where one model consistently outperforms the other, using controlled randomness to ensure reproducibility.



### 1. Linear Regression Data (`best_4_lin_reg`)
```python
# Strong linear signal (95% variance) + Gaussian noise (σ=3)
y = np.dot(X, coefficients) + np.random.normal(0, 3, X.shape[0])
# Hidden non-linear test
y += 0.01 * X[:, 0]**2 - 0.005 * (X[:, 1] * X[:, 2])
```

### 2. Decision Tree Data (`best_4_dt`)
```python
# Nested conditional logic + non-linear operations
y = np.log(X[:, 0]) + np.sin(X[:, 1]) + (X[:, 2]**2) * (X[:, 3] > 0.5)
# Minimal noise (σ=0.1)
y += np.random.normal(0, 0.1, X.shape[0])
```

---

## 📊 Performance Metrics

| Metric                | Linear Data | Tree Data |
|-----------------------|-------------|-----------|
| Win Rate              | 86.7%       | 80.0%     |
| Avg RMSE Ratio        | 0.44×       | 0.15×     |
| Max Features          | 5           | 10        |

---

## 🛠️ Engineering Highlights

### Seed Control
```python
np.random.seed(seed)  # Full reproducibility
assert np.all(data1 == data2)  # Consistency check
```

### Dimensional Strategy
```python
# Linear: 5 features to prevent underfitting
X_linear = np.random.rand(1000, 5)

# Tree: 10 features for complex splits
X_tree = np.random.rand(1000, 10)
```

---

moral of the story, quality data = quality insights ! 
