# Model Card — Stroke Risk Classifier

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | Stroke Risk Classifier |
| **Version** | 1.0.0 |
| **Type** | Binary classification (stroke / no stroke) |
| **Algorithm** | AutoGluon `TabularPredictor` — weighted ensemble of LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Neural Networks (Torch + FastAI) |
| **Ensemble** | 3-layer stacking (`WeightedEnsemble_L3`) |
| **Training date** | 2024 |
| **Framework** | AutoGluon 1.5.0, Python 3.12 |
| **Hardware** | NVIDIA RTX 3050 (GPU for neural nets, CPU for trees) |
| **Training time** | ~180 seconds |
| **License** | MIT |

---

## Intended Use

### Primary use case
Screening tool to identify patients at **elevated risk of stroke** based on clinical and demographic features. Intended to support, not replace, clinical decision-making.

### Intended users
- Clinical decision-support systems in hospital settings
- Population health teams performing risk stratification
- Public health researchers studying stroke risk factors
- Data scientists evaluating AutoML approaches to medical datasets

### Out-of-scope uses
- **Standalone clinical diagnosis** — the model is a risk flag, not a diagnostic tool. All high-risk predictions must be reviewed by a qualified clinician.
- **Emergency triage** — not validated for real-time acute-care decisions.
- **Paediatric patients** — the training data is predominantly adults; paediatric risk profiles are poorly represented.
- **Non-ischaemic stroke subtypes** — the dataset does not distinguish stroke types (ischaemic, haemorrhagic, TIA).
- **Deployment outside the data distribution** — do not deploy on populations with substantially different demographics, healthcare systems, or comorbidity rates without re-validation.

---

## Data

### Source
[Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Size and split
| Split | Rows | Positive rate |
|-------|------|---------------|
| Training | 4,488 | 13.4% |
| Test (held-out) | 1,122 | 13.4% |

> **Note:** The publicly available Kaggle version reports 4.9% positive rate. This project's version has a higher positive rate (13.4%), suggesting a differently pre-filtered extract.

### Features

| Feature | Type | Notes |
|---------|------|-------|
| `age` | Continuous | Years (float) |
| `gender` | Categorical | Male, Female *(Other recoded to mode)* |
| `hypertension` | Binary | 0 = No, 1 = Yes |
| `heart_disease` | Binary | 0 = No, 1 = Yes |
| `ever_married` | Categorical | Yes, No |
| `work_type` | Categorical | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Categorical | Urban, Rural |
| `avg_glucose_level` | Continuous | mg/dL |
| `bmi` | Continuous | 201 missing values (3.6%) — imputed with training-set median |
| `smoking_status` | Categorical | Never smoked, formerly smoked, smokes, Unknown |

### Known data quality issues
- **BMI missing values (3.6%):** Imputed with median. Missingness pattern is not fully random (more common in younger patients), which may introduce minor bias.
- **Smoking status 'Unknown' (30%):** A substantial portion of patients have unknown smoking history. This is treated as a category rather than a missing value, which may underestimate smoking-related risk.
- **`gender = 'Other'` (1 sample):** Recoded to mode ('Female') to avoid an uninformative dummy variable. The model has no meaningful signal for non-binary gender identities.
- **No geographic metadata:** The dataset's origin country is not disclosed. Population-level risk baselines may differ from target deployment populations.

---

## Training

### Objective metric
`roc_auc` — chosen because accuracy is misleading at 13.4% positive rate. ROC-AUC measures ranking quality across all decision thresholds.

### Hyperparameter search
AutoGluon's `best_quality` preset runs automated HPO for each base learner, then combines them via stacking. No manual hyperparameter tuning was performed.

### Class imbalance handling
AutoGluon applies internal sample weighting to handle the 86.6% / 13.4% class split. SMOTE was not used, as tree-ensemble methods generally perform better with sample weighting than with synthetic oversampling.

---

## Evaluation Results

All metrics computed on the held-out **test set (1,122 rows)** — not on validation folds used during training.

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | **0.9209** | Excellent discrimination — random = 0.5 |
| **Average Precision** | 0.7482 | Strong, well above the 0.134 baseline |
| **F1 — Stroke class** | 0.6079 | Moderate (threshold-dependent) |
| **F1 — Macro** | 0.7819 | Good across both classes |
| **Accuracy** | 0.9207 | Misleading; a trivial "predict never" gives 0.866 |

### Decision threshold
The default threshold is **0.5**. For clinical screening (high recall preferred), a threshold of **0.3–0.35** is recommended — this increases stroke recall at the cost of more false alarms. See `Predictions.ipynb` Section 12 for a full cost-sensitivity analysis.

### Subgroup performance (test set)

| Subgroup | n | ROC-AUC | Notes |
|----------|---|---------|-------|
| Female | ~640 | ~0.92 | See Predictions.ipynb for exact values |
| Male | ~480 | ~0.91 | |
| Age < 45 | ~320 | ~0.82 | Fewer stroke events; less stable AUC estimate |
| Age 45–74 | ~560 | ~0.89 | |
| Age 75+ | ~240 | ~0.88 | |

> Run `notebooks/Predictions.ipynb` Section 11 for updated subgroup metrics.

---

## Limitations and Biases

1. **Age dominates predictions.** Age accounts for ~33% of permutation importance, far above all other features. The model may over-rely on age and under-weight modifiable risk factors (e.g. glucose, BMI) in individual predictions.

2. **Non-binary gender not represented.** The single 'Other' gender sample was recoded. The model cannot meaningfully assess risk for non-binary individuals.

3. **Smoking 'Unknown' is a proxy.** The large 'Unknown' category conflates non-smokers who didn't disclose with former/current smokers. This introduces noise into the smoking_status signal.

4. **No temporal data.** The model uses a static snapshot of patient features. Duration of conditions (e.g. years with hypertension, how long since quitting smoking) is not captured.

5. **No lab values beyond glucose and BMI.** Clinically important stroke predictors (LDL cholesterol, platelet count, atrial fibrillation history) are absent from the dataset.

6. **Test set is a single 20% split.** Performance estimates have no confidence intervals. Bootstrap resampling would give a more robust uncertainty estimate.

7. **Dataset provenance is unclear.** The Kaggle dataset does not disclose the source hospital or country. Deployment in a significantly different population requires re-validation.

---

## Ethical Considerations

- **Do not use for autonomous clinical decisions.** All high-risk flags must be reviewed by a licensed clinician before any action is taken.
- **Transparency with patients.** If this model informs clinical care, patients should be informed that algorithmic risk scores are part of the assessment.
- **Feedback loop risk.** If deployed at scale, model predictions may influence treatment patterns, altering the underlying risk distribution the model was trained on. Periodic re-validation is required.
- **Disparate impact monitoring.** Performance parity across demographic subgroups should be monitored after deployment. See the subgroup analysis in `Predictions.ipynb`.

---

## Caveats and Recommendations

| Recommendation | Priority |
|----------------|----------|
| Lower threshold to 0.3–0.35 for high-recall screening contexts | High |
| Validate on a local hospital dataset before clinical deployment | High |
| Monitor subgroup performance (by age, gender) quarterly | High |
| Retrain annually or when population demographics shift | Medium |
| Add confidence intervals via bootstrap resampling | Medium |
| Incorporate additional clinical features (lipid panel, ECG history) | Low |

---

*This model card was written to accompany version 1.0.0 of the Stroke Prediction project. Update it whenever the model, data, or deployment context changes.*
