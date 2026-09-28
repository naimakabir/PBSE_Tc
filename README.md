# PBSE-Tc: Physics-Based Stacking Ensemble for Superconducting Critical Temperature Prediction

This repository contains the code and pipeline for **PBSE**, a Physics-Based Stacking Ensemble that predicts the superconducting critical temperature (T<sub>c</sub>) of compounds using both compositional descriptors and physics-derived features grounded in BCS and Ginzburg-Landau (GL) theory.

The model augments the standard 81 compositional statistics used in prior T<sub>c</sub> prediction work with two theory-derived quantities, coherence length (ξ) and upper critical field (H<sub>c2</sub>), estimated from a predicted Fermi velocity and a leakage-free proxy for T<sub>c</sub>. These physics-informed features rank above all 81 compositional descriptors in SHAP importance and improve prediction accuracy over composition-only baselines.

## Paper

Naima Kabir and Mohammad Rafiqul Islam, *A Physics-Based Stacking Ensemble for Predicting Superconducting Critical Temperature Using BCS-Ginzburg-Landau Derived Features*, University of Chittagong.

## Key results

| Evaluation | R² | RMSE (K) | MAE (K) |
|---|---|---|---|
| UCI held-out test set (21,263 compounds) | 0.938 | 8.48 | 4.70 |
| External validation, Stanev dataset (250 disjoint compounds) | 0.810 | 14.87 | 7.77 |

External validation compounds were confirmed non-overlapping with the UCI training set. Coherence length and upper critical field for the external set were derived from a Tc-proxy model trained only on UCI data, never from measured critical temperature, to avoid leaking the prediction target into the input features.

## Pipeline overview

The pipeline runs in five stages:

1. **Fermi velocity (v<sub>F</sub>) estimation** — A 162-material reference dataset spanning six superconducting families is used to build a hybrid classification-and-lookup system: chemically unambiguous compounds are assigned v<sub>F</sub> from literature-sourced canonical values by subfamily, and compounds falling into catch-all categories are estimated by a tuned XGBoost regressor.
2. **Out-of-fold T<sub>c</sub> estimation** — To derive physics features without leaking the true target, an XGBoost model generates 10-fold out-of-fold T<sub>c</sub> predictions on the full UCI dataset. These, not the measured T<sub>c</sub>, are used downstream.
3. **BCS-GL feature derivation** — Coherence length (ξ) and upper critical field (H<sub>c2</sub>) are computed from the predicted v<sub>F</sub> and the out-of-fold T<sub>c</sub> using standard BCS clean-limit and Ginzburg-Landau relations.
4. **SHAP-based feature selection** — A preliminary XGBoost model ranks all 84 candidate features (81 compositional plus v<sub>F</sub>, ξ, H<sub>c2</sub>) by mean absolute SHAP value; the top 30 are retained for the final ensemble.
5. **Stacking ensemble** — Five tuned base learners (KNN, SVR, MLP, Random Forest, LightGBM) are combined through a Ridge meta-learner using scikit-learn's `StackingRegressor`.

External validation on the Stanev dataset follows the same feature pipeline, with v<sub>F</sub>, ξ, and H<sub>c2</sub> derived independently using a Tc-proxy model trained exclusively on UCI data.

## Repository structure

```
├── data/
│   ├── train_raw.csv                     # UCI dataset (81 compositional features for 21,263 compounds)
│   ├── unique_m.csv                      # UCI dataset (compositions of 21,263 compound)
│   ├── Fermi_169_unique_v5.csv           # Fermi velocity reference dataset (162 materials)
│   └── stanev_250_external.csv           # External validation set (Stanev, disjoint from UCI)
├── notebooks/
│   ├── fermi_velocity_162.ipynb          # Stage 1: pick features from UCI dataset for training vF dataset.
│   ├── vF_xi_Hc2_derivation.ipynb        # Stages 2-4: vF classification and regression pipeline, OOF Tc estimation, GL feature derivation
│   ├── shap_feature_selection.ipynb      # Stage 5: SHAP ranking and top-30 selection
│   ├── stacking_ensemble.ipynb           # Stage 6: base learner tuning and stacking ensemble
│   ├── external_validation.ipynb         # Stanev external validation, leakage-free Tc-proxy
│   ├── per_family_breakdown.ipynb        # Per-family test-set performance (Table 6)
│   ├── paired_significant_test.ipynb     # Paired t-test and Wilcoxon tests, stacking vs. base learners (Table 5)
│   └── Tc_stratified_analysis.ipynb      # Error, bias, and within-tolerance rates by Tc range (Table 7)
├── models/
│   ├── vF_hybrid_model.pkl               # Trained Fermi velocity fallback model (can get upon running the vF_xi_Hc2_derivation.ipynb file)
│   ├── vF_family_encoder.pkl             # One-hot encoder for family labels
│   └── stacking_model_30features.pkl     # Final trained stacking ensemble (from stacking_ensemble.ipynb)
├── requirements.txt
└── README.md
```

## Requirements

```
python >= 3.9
numpy
pandas
scikit-learn
xgboost
lightgbm
shap
scipy
matplotlib
joblib
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Run the notebooks in order:

```bash
jupyter notebook notebooks/fermi_velocity_162.ipynb
jupyter notebook notebooks/vF_xi_Hc2_derivation.ipynb
jupyter notebook notebooks/SHAP_feature_selection.ipynb
jupyter notebook notebooks/stacking_ensemble.ipynb
jupyter notebook notebooks/external_validation.ipynb
jupyter notebook notebooks/paired_significant_test.ipynb
jupyter notebook notebooks/per_family_breakdown.ipynb
jupyter notebook notebooks/Tc_stratified_analysis.ipynb
```

Each notebook reads its inputs from `data/` and writes intermediate outputs back to the same directory. Trained model artifacts are saved to `models/`.

## Data sources

- **UCI Superconductivity Dataset**: Hamidieh, K. (2018). *A data-driven statistical model for predicting the critical temperature of a superconductor*. Computational Materials Science, 154, 346-354. Originally compiled from the SuperCon database maintained by the Japanese National Institute for Materials Science (NIMS).
- **External validation set**: Stanev, V. et al. (2018). *Machine learning modeling of superconducting critical temperature*. npj Computational Materials, 4, 29.

## Methodological notes on leakage avoidance

Because coherence length and upper critical field are deterministic functions of T<sub>c</sub> and Fermi velocity, computing them from the measured T<sub>c</sub> would leak the prediction target directly into the model's input features. This pipeline avoids that at every stage:

- On the UCI training set, ξ and H<sub>c2</sub> are derived from **out-of-fold** T<sub>c</sub> predictions, never from the measured value.
- On the external Stanev set, ξ and H<sub>c2</sub> are derived from a **Tc-proxy model trained only on UCI data**, with hyperparameters selected using a held-out UCI split, entirely independent of the Stanev compounds' measured T<sub>c</sub>.

## License

MIT License. See `LICENSE` for details.

## Citation

If you use this code or pipeline, please cite:

```
Kabir, N. and Islam, M. R. A Physics-Based Stacking Ensemble for Predicting
Superconducting Critical Temperature Using BCS-Ginzburg-Landau Derived
Features. University of Chittagong, 2026.
```

## Contact

Naima Kabir, Department of Physics, University of Chittagong, Chattogram-4331, Bangladesh.
