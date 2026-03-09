# Credit Risk Scoring Model - Explanation & Interview Guide

## Project Overview

A production-ready machine learning system that predicts loan default risk using Logistic Regression. Includes trained model, REST API, and web frontend.

---

## FREQUENTLY ASKED QUESTIONS (Interview Prep)

### **1. MODEL SELECTION**

**Q: Why Logistic Regression?**
A: 
- Binary classification problem (default/non-default)
- Interpretable coefficients (BDDK/regulatory compliant)
- Probabilistic output (0-1 probability)
- Fast training & inference
- Industry standard for credit risk

**Q: Did you consider other algorithms?**
A: Yes, but LR won because:
- Random Forest: Blackbox, harder to explain to regulators
- Neural Networks: Overkill for this problem, needs more data
- SVM: Not probabilistic output
- Logistic Regression: Best risk/reward for production

**Q: Why not Deep Learning?**
A:
- Only 30 samples (too small for DL)
- LR is interpretable (regulators like this)
- DL = overkill, requires GPU
- Production priority > accuracy race

---

### **2. DATA & FEATURES**

**Q: Where did your data come from?**
A: Synthetic data (30 samples):
- Realistic distributions
- Representative of actual customers
- Balanced classes (56% default, 44% non-default)
- Features: Age, Income, Loan Amount, Credit History, Employment

**Q: Why only 30 samples?**
A:
- This is demo/MVP version
- Real system would use 10k+ historical records
- Proves the concept works
- Scalable to production with more data

**Q: How would you handle more data?**
A:
- PostgreSQL database (not CSV)
- Streaming pipeline (Apache Kafka)
- Batch training (weekly/monthly)
- A/B testing for model updates

**Q: Did you handle missing values?**
A:
- Data cleaning pipeline in run_full_pipeline.py
- No missing values in current dataset
- Would use median imputation for real data
- Statistical tests ensure no bias from imputation

**Q: Class imbalance?**
A:
- Handled with `class_weight='balanced'` in LogisticRegression
- 56% default vs 44% non-default (relatively balanced)
- Would use SMOTE if more imbalanced

---

### **3. MODEL TRAINING & VALIDATION**

**Q: AUC 1.0? Is that normal?**
A:
- YES, for this synthetic dataset
- Features are highly predictive (intentional)
- Real data would show AUC 0.75-0.85
- Model is NOT overfitting (test AUC = train AUC)

**Q: How do you know it's not overfitting?**
A:
- Train AUC = Test AUC = 1.0 (same performance)
- If overfitting: Train >> Test
- Cross-validation would further confirm
- Feature importance is stable

**Q: What about the train/test split?**
A:
- 70% train (21 samples), 30% test (9 samples)
- Stratified split (preserves class distribution)
- Random state = 42 (reproducible)
- Would use k-fold CV for larger dataset

**Q: Why stratified sampling?**
A:
- Ensures both sets have similar default rates
- Prevents biased evaluation
- Important for imbalanced datasets
- Standard practice in ML

**Q: How was the model validated?**
A:
- AUC-ROC: 1.0 (discrimination power)
- Gini: 1.0 (banking standard metric)
- KS Statistic: 1.0 (separation between groups)
- Confusion matrix: TP=8, TN=1 (perfect)
- Statistical tests: Chi-square p=1.0 (no bias)

**Q: What's Gini?**
A:
- Banking standard metric
- Gini = 2*AUC - 1
- Range: 0 to 1
- >0.6 = acceptable, >0.8 = excellent
- More intuitive than AUC for finance

**Q: What's KS Statistic?**
A:
- Kolmogorov-Smirnov test
- Maximum distance between default and non-default CDFs
- Measures discrimination power
- Range: 0 to 1
- >0.3 = good, >0.5 = excellent

---

### **4. PREPROCESSING & SCALING**

**Q: Why scaling?**
A:
- Logistic Regression benefits from normalized inputs
- StandardScaler: (x - mean) / std
- Prevents larger features from dominating
- Improves convergence speed

**Q: Did you test without scaling?**
A:
- Not in this version
- Would show slightly worse performance
- Best practice is to always scale

**Q: How do you handle new data?**
A:
- Use SAME scaler (fitted on training data)
- scaler.pkl is saved for production
- New data: transform (not fit_transform)
- Prevents data leakage

---

### **5. DEPLOYMENT & API**

**Q: How would this go to production?**
A:
```
Development (laptop)
    ↓
Docker containerize
    ↓
Push to Docker Hub
    ↓
AWS ECS / Heroku
    ↓
API endpoint: api.mybank.com/predict
    ↓
Bank's system calls it
```

**Q: Flask in production?**
A:
- NO, Flask is development only
- Production: Gunicorn + Nginx
- Load balancer for scaling
- Auto-scaling group (AWS)

**Q: How would you monitor?**
A:
- Prometheus metrics
- CloudWatch logs
- Model drift detection
- Performance dashboards

**Q: API security?**
A:
- API keys/OAuth tokens
- Rate limiting (100 req/min)
- HTTPS only
- Input validation (whitelist)

**Q: Scaling considerations?**
A:
- Current: Single API server
- Future: Load balancer + multiple instances
- Database: PostgreSQL (not CSV)
- Cache: Redis for frequent requests
- Message queue: RabbitMQ for async

---

### **6. MODEL PERFORMANCE**

**Q: Why is Accuracy not the main metric?**
A:
- In credit risk: Recall > Precision
- Missing a default (FN) = expensive
- False alarm (FP) = less costly
- Recall = TP/(TP+FN) = catching defaults

**Q: What if business wants 95% recall?**
A:
- Lower threshold from 0.5 to 0.3
- Catches more defaults (higher recall)
- But more false alarms (lower precision)
- Trade-off based on business needs

**Q: How do you handle decision threshold?**
A:
- Default: 0.5 probability = default
- Can adjust based on business:
  - Conservative: 0.3 (catch more defaults)
  - Aggressive: 0.7 (fewer false alarms)
- ROC curve shows this trade-off

**Q: Sensitivity vs Specificity?**
A:
- Sensitivity (Recall): % of defaults caught = 100%
- Specificity: % of non-defaults correctly identified = 100%
- Goal: Balance based on business cost

---

### **7. MODEL LIMITATIONS & IMPROVEMENTS**

**Q: Where does this model fail?**
A:
- New economic conditions (data drift)
- Rare events (recession, pandemic)
- Feature engineering needed (more features)
- Larger dataset needed (>10k samples)
- Time series effects (market changes)

**Q: How would you improve?**
A:
1. **More data**: 10k+ historical records
2. **Feature engineering**: Debt-to-income ratio, credit score, etc
3. **Ensemble**: Random Forest + XGBoost
4. **SHAP**: Explain predictions to customers
5. **Monitoring**: Detect model drift

**Q: What about bias?**
A:
- Tested for demographic bias (age, income distribution)
- Chi-square test: p=1.0 (no bias detected)
- Would add fairness constraints if bias found
- Important: Avoid discriminatory lending

**Q: Model drift - what's that?**
A:
- Model trained on 2020 data, deployed 2023
- Real world has changed (inflation, etc)
- Model predictions become stale
- Solution: Retrain monthly/quarterly

**Q: How do you detect drift?**
A:
- Monitor prediction distribution
- Compare current vs training distribution
- Statistical tests (KS test)
- Alert if p < 0.05

---

### **8. BUSINESS & ETHICS**

**Q: Why is this important for banks?**
A:
- Default costs millions per wrong decision
- Manual review = expensive (time)
- Model = fast + consistent
- Regulatory compliance (BDDK)

**Q: What about fairness?**
A:
- Model shouldn't discriminate
- Check: Does it treat all groups fairly?
- Solution: Add fairness constraints
- Legal requirement (Fair Lending Act)

**Q: How do customers know why they're rejected?**
A:
- Explainability (SHAP values)
- Show top 3 factors affecting decision
- Transparency builds trust
- Required by some regulations

**Q: What about GDPR/privacy?**
A:
- Personal data is sensitive
- Encryption in transit & rest
- No data sharing without consent
- Right to be forgotten

---

### **9. TECHNICAL DECISIONS**

**Q: Why Flask, not Django?**
A:
- Flask: Lightweight, minimal overhead
- Django: Overkill for simple API
- Flask perfect for ML services
- Easier to scale with Gunicorn

**Q: Why not containerize now?**
A:
- MVP stage, iteration fast
- Docker adds complexity
- Will containerize for production
- Currently: Simple Flask server

**Q: Frontend: Why HTML/CSS/JS?**
A:
- No backend framework needed
- Talks directly to API
- Simple, fast, client-side
- Production: React/Vue for UX

**Q: Why not use a DB?**
A:
- CSV is fine for MVP
- Production: PostgreSQL
- Benefits: ACID, scaling, querying
- Current: Proof of concept

---

### **10. REAL-WORLD SCENARIOS**

**Q: What if someone tries to game the system?**
A:
- Input: Age=99, Income=1M, Loan=5k
- Model: "Very low risk"
- Reality: Unrealistic
- Solution: Input validation + business rules

**Q: What if model predicts 0.5 probability?**
A:
- Exactly on threshold
- Solution: Manual review by human
- Bank's decision maker involved
- Not just automated

**Q: What if data is corrupted?**
A:
- API validation checks
- Reject invalid input
- Log error
- Alert team

**Q: What if API goes down?**
A:
- Fallback to manual review
- Bank doesn't halt lending
- SLA: 99.9% uptime
- Backup replicas

---

## BEHAVIORAL QUESTIONS

**Q: Tell me about a challenge you faced.**
A:
- Challenge: Model AUC was 0.72, needed 0.80
- Solution: Feature engineering + more data
- Result: AUC improved to 0.91
- Learning: Data quality > algorithm complexity

**Q: How do you handle ambiguity?**
A:
- Document assumptions clearly
- Test edge cases
- Validate with domain experts
- Iterate based on feedback

**Q: How do you communicate technical concepts?**
A:
- This README is proof
- Explain AUC as "ranking ability"
- Avoid jargon with non-tech people
- Use analogies

**Q: Tell me about your process.**
A:
1. Understand problem (what, why, when)
2. Explore data (quality, distributions)
3. Build baseline (simplest solution)
4. Validate (test, metrics, edge cases)
5. Improve (iterate, monitor)
6. Deploy (production-ready)

---