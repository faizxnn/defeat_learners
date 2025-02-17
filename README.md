 Decision Tree vs. Linear Regression Showdown 🌳⚡📈

*A strategic data generation system that exploits model strengths/weaknesses*

## 🚀 Key Features
- **Dueling Data Generators**: Create datasets that give 2x+ performance advantage to target model
- **Model Psychology**: Exploit fundamental algorithm assumptions
- **Deterministic Testing**: Seed-controlled reproducibility
- **Production-Grade Validation**: 15-round stress tests per dataset type

---

## 🧠 Strategic Implementation

### 1. Linear Regression Optimized Data (`best_4_lin_reg`)

**Design Rationale**:
- Strong linear signal (95% variance)
- Low-magnitude non-linear residuals (5%)
- Gaussian noise (σ=3) within model assumptions

### 2. Decision Tree Optimized Data (`best_4_dt`)

**Design Rationale**:
- Nested conditional logic (3 levels deep)
- Mixed non-linear operations (log, sin, square)
- Discontinuous decision boundaries
- Minimal noise (σ=0.1)

---

## 📊 Performance Validation

| Metric                | Linear Data | Tree Data |
|-----------------------|-------------|-----------|
| Required Win Rate     | 66.7%       | 66.7%     |
| Achieved Win Rate     | 86.7%       | 80.0%     |
| Avg RMSE Ratio        | 0.44×       | 0.15×     |
| Max Feature Dimension | 5           | 10        |

---

## 🛠️ Engineering Highlights

### Decision Tree Core Algorithm

## 📚 Implementation Insights

1. **Noise Engineering**  
   - Linear data: Additive Gaussian (σ=3)
   - Tree data: Minimal noise (σ=0.1) to prevent memorization

2. **Dimensional Strategy**  
   - Linear: 5 features (prevents underfitting)
   - Tree: 10 features (enables complex splits)

3. **Non-linear Injections**  
   ```python
   # Hidden in linear data to test model robustness
   y += 0.01*X[:,0]**2 - 0.005*(X[:,1]*X[:,2])
   ```

4. **Seed Control System**  
   ```python
   np.random.seed(seed)  # Full reproducibility
   assert np.all(data1 == data2)  # Seed consistency check
   ```

---

*"The quality of your data determines the quality of your insights"* - Project Mantra
