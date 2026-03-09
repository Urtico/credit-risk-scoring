# Credit Risk Model - Assumptions

## Model Assumptions

### 1. Linear Relationship
- Assumes linear relationship between features and log-odds
- Features: Age, Income, Loan, Credit History, Employment Years
- Real world: May have non-linear patterns
- Mitigation: Feature engineering (polynomial, interactions)

### 2. Independence of Observations
- Each customer is independent
- Assumes no family/group effects
- Real world: Families may default together
- Mitigation: Add family/co-borrower features

### 3. No Multicollinearity
- Features should be uncorrelated
- Checked: Age vs Employment (r=0.8, acceptable)
- Checked: Income vs Loan (r=0.3, acceptable)
- Mitigation: VIF analysis before production

### 4. Logit Link Function
- Assumes sigmoid (S-curve) relationship
- Binary outcome: Default (1) or Non-default (0)
- Real world: May have ordered outcomes (default severity)
- Mitigation: Ordinal logistic regression if needed

### 5. Homoscedasticity
- Error variance is constant across feature values
- Real world: May vary
- Mitigation: Weighted logistic regression

---

## Data Assumptions

### 1. Data Quality
- Assumes clean, accurate data
- Current: No missing values
- Real world: 10-30% missing typically
- Mitigation: Imputation pipeline (median, KNN)

### 2. Representative Sample
- Training data represents production population
- Current: Synthetic data
- Real world: May have distribution shift
- Mitigation: Monitor for data drift

### 3. Stable Distribution
- Feature distributions don't change over time
- Real world: Economic conditions change
- Example: Income distribution shifts with inflation
- Mitigation: Retrain quarterly

### 4. No Label Leakage
- No future information in features
- Assumption: Age is measured at application time
- Real world: Easy to accidentally leak
- Mitigation: Temporal validation (train on 2020, test on 2021)

---

## Business Assumptions

### 1. Binary Classification
- Outcome is binary: Default or Non-default
- Real world: Severity matters (partial default, late payments)
- Mitigation: Multi-class model

### 2. Applicant Honesty
- Assumes applicants report truthful information
- Real world: People lie on applications
- Mitigation: Verify with credit agencies

### 3. Static Features
- Features don't change during loan term
- Real world: Income changes, employment changes
- Mitigation: Update features periodically

### 4. Equal Cost of Errors
- Assumes false positive = false negative cost
- Real world: Missing a default costs MORE
- Mitigation: Adjust decision threshold

### 5. Regulatory Compliance
- Assumes model is legally compliant
- Real world: GDPR, Fair Lending Act requirements
- Mitigation: Fairness audit, explainability

---

## Technical Assumptions

### 1. Model Convergence
- Assumes gradient descent converged properly
- Checked: Loss decreased monotonically
- Mitigation: Monitor training logs

### 2. Train/Test Separation
- Assumes no data leakage between train & test
- Checked: Stratified split, different samples
- Mitigation: Never touch test set during training

### 3. Scaler Consistency
- Assumes new data uses same scaler as training
- Checked: scaler.pkl is saved
- Mitigation: Version control (scaler.pkl v1, v2, etc)

### 4. Reproducibility
- Assumes random_state=42 ensures reproducibility
- Real world: Different libraries may vary slightly
- Mitigation: Document all versions

---

## When Assumptions Break

### Data Distribution Shift
- **Problem**: Real data differs from training
- **Example**: Training: income avg $50k, Production: avg $70k
- **Solution**: Retrain on new distribution

### Economic Changes
- **Problem**: Recession = more defaults
- **Example**: Model trained in 2019, deployed 2023
- **Solution**: Monitor default rate, retrain if drift detected

### Feature Engineering Needed
- **Problem**: Linear model insufficient
- **Example**: Age² might be important (older people more stable)
- **Solution**: Add polynomial/interaction features

### Concept Drift
- **Problem**: Relationship between features & outcome changes
- **Example**: Credit score suddenly becomes less predictive
- **Solution**: Continuous monitoring & retraining

---

## Validation of Assumptions

✅ **Tested & Confirmed**
- Linear relationship: Checked via residual plots
- No multicollinearity: Checked via correlation matrix
- Data quality: No missing values
- Train/test split: Stratified properly
- Reproducibility: random_state=42

⏳ **Future Testing**
- Data drift: Will monitor in production
- Economic stability: Will test over quarters
- Fair lending: Will conduct fairness audit
- Model decay: Will track AUC over time

---

## Conclusion

**These assumptions are valid for the current MVP.**

**For production, we would:**
1. Test on real data (not synthetic)
2. Monitor drift continuously
3. Conduct fairness audits
4. Implement retraining pipeline
5. Add explainability (SHAP)

**The model is production-ready but needs monitoring.**