# MLflow Tracking for Anomaly Detection

This project demonstrates how to use **MLflow** to track and compare machine learning experiments in an **Anomaly Detection** scenario. We use various machine learning models to classify a highly imbalanced dataset and evaluate their performance. The MLflow platform allows us to efficiently log, track, and analyze experiments, ensuring reproducibility and transparency throughout the machine learning lifecycle.

---

## 📋 Problem Statement

The dataset is highly imbalanced:
- **Class 0 (Majority Class):** 900 samples
- **Class 1 (Minority Class):** 100 samples

The objective is to maximize the **recall for the minority class (Class 1)** while maintaining acceptable performance on the majority class (Class 0). This setup simulates real-world use cases such as fraud detection or anomaly detection, where the minority class often holds more importance.

---

## 🛠️ Workflow

### 1️⃣ **Dataset Preparation**
- A synthetic dataset was created using `scikit-learn`'s `make_classification()` function with imbalanced class weights.

### 2️⃣ **Model Training**
We trained four models:
1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**
4. **XGBoost with SMOTETomek** (oversampling + undersampling to handle imbalance)

### 3️⃣ **Experiment Tracking with MLflow**
- MLflow tracks:
  - **Parameters:** Hyperparameters of the models (e.g., learning rate, max depth).
  - **Metrics:** Performance metrics (e.g., accuracy, recall, F1 score).
  - **Artifacts:** Trained models, environment files, and other outputs.
  
---

## 🖥️ Key Features of MLflow

1. **Experiment Logging:** Each model run is logged as an independent experiment in MLflow.
2. **Metrics Comparison:** Metrics like `accuracy`, `recall_class_0`, `recall_class_1`, and `f1_score_macro` are compared across models.
3. **Artifact Management:** Models and their dependencies are stored as artifacts for reproducibility.
4. **Visualization:** MLflow UI provides plots for model comparison, such as **Parallel Coordinates Plot**.

---

## 🔍 Analysis of Results

| **Model**                      | **Accuracy** | **F1 Score (Macro)** | **Recall (Class 0)** | **Recall (Class 1)** |
|---------------------------------|--------------|-----------------------|----------------------|----------------------|
| **Logistic Regression**         | 0.917        | 0.750                 | 0.963                | 0.500                |
| **Random Forest**               | 0.963        | 0.882                 | 0.996                | 0.667                |
| **XGBoost**                     | 0.977        | 0.930                 | 0.996                | 0.800                |
| **XGBoost with SMOTETomek**     | 0.963        | 0.900                 | 0.978                | 0.833                |

### Observations:
1. **Logistic Regression:** Struggles with Class 1 recall (50%), despite acceptable accuracy.
2. **Random Forest:** Improves Class 1 recall (66.7%), but still suboptimal.
3. **XGBoost:** Delivers balanced performance with a high recall for both classes (80% for Class 1).
4. **XGBoost with SMOTETomek:** Achieves the **highest recall for Class 1 (83.3%)**, which is crucial for anomaly detection. However, there is a slight trade-off in precision.

---

## 📊 Visualization

### Parallel Coordinates Plot
![Parallel Coordinates Plot](path/to/your/parallel_coordinates_plot.png)

The **Parallel Coordinates Plot** visualizes the trade-offs between metrics for different models:
- XGBoost models (with and without SMOTETomek) show the best balance between **recall** and **F1 score**.
- Logistic Regression shows poor performance for Class 1 recall.

---

## 🏆 Best Model

The **XGBoost with SMOTETomek** model is the best choice for this problem:
- **Strengths:** Highest recall for Class 1 (minority class) with balanced overall performance.
- **Artifacts:** Saved in MLflow under the run name `XGBClassifier With SMOTE`.

Artifact Files:
- **model.xgb**: Trained model file.
- **conda.yaml**: Environment file for reproducibility.
- **python_env.yaml**: Python environment specification.
- **requirements.txt**: Dependencies for deployment.

---

## 🚀 Future Work

1. **Model Deployment:** Utilize MLflow’s Model Registry to deploy the best model into production.
2. **Advanced Monitoring:** Integrate with monitoring tools (e.g., Prometheus) to track production model performance.
3. **Pipeline Automation:** Use CI/CD pipelines to automate model training, tracking, and deployment.

---

## 🤝 Why MLflow?

1. **Reproducibility:** Tracks all parameters, metrics, and artifacts for easy replication.
2. **Collaboration:** Provides a shared platform for teams to collaborate and share results.
3. **Scalability:** Integrates seamlessly into MLOps pipelines, supporting end-to-end workflows.

---

## 💡 How to Use This Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/mlflow-anomaly-detection.git
   cd mlflow-anomaly-detection

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Start MLflow UI:
   ```bash
   mlflow ui

Open http://127.0.0.1:5000 in your browser.

4. Run the notebook:
   ```bash
   jupyter notebook MLflow.ipynb
   
5. Visualize results and compare models in the MLflow UI.
   

## 🔗 References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

