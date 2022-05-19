# modelop.schema.0: input_schema.avsc
# modelop.schema.1: output_schema.avsc

import pandas as pd
import pickle
import numpy as np

# Bias libraries
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.bias import Bias 


# modelop.init
def begin():
    
    global logreg_classifier
    
    # load pickled logistic regression model
    logreg_classifier = pickle.load(open("logreg_classifier.pickle", "rb"))

    
# modelop.score
def action(data):
    
    # Turn data into DataFrame
    data = pd.DataFrame([data])
    
    # There are only two unique values in data.number_people_liable.
    # Treat it as a categorical feature
    data.number_people_liable = data.number_people_liable.astype('object')

    predictive_features = [
        'duration_months', 'credit_amount', 'installment_rate',
        'present_residence_since', 'age_years', 'number_existing_credits',
        'checking_status', 'credit_history', 'purpose', 'savings_account',
        'present_employment_since', 'debtors_guarantors', 'property',
        'installment_plans', 'housing', 'job', 'number_people_liable',
        'telephone', 'foreign_worker'
    ]
    
    data["score"] = logreg_classifier.predict(data[predictive_features])
    
    # MOC expects the action function to be a *yield* function
    yield data.to_dict(orient="records")


# modelop.metrics
def metrics(df_baseline, data):
    # dictionary to hold final metrics
    metrics = {}

    # convert data into DataFrame
    data = pd.DataFrame(data)

    # getting dummies for shap values
    data_processed = preprocess(data)[predictive_features]

    # calculate metrics
    f1 = f1_score(data["label_value"], data["score"])
    cm = confusion_matrix(data["label_value"], data["score"])
    labels = ["Default", "Pay Off"]
    cm = matrix_to_dicts(cm, labels)
    fpr, tpr, thres = roc_curve(data["label_value"], data["predicted_probs"])
    auc_val = roc_auc_score(data["label_value"], data["predicted_probs"])
    roc = [{"fpr": x[0], "tpr": x[1]} for x in list(zip(fpr, tpr))]

    # assigning metrics to output dictionary
    metrics["performance"] = [
        {
            "test_name": "Classification Metrics",
            "test_category": "performance",
            "test_type": "classification_metrics",
            "test_id": "performance_classification_metrics",
            "values": {"f1_score": f1, "auc": auc_val, "confusion_matrix": cm},
        }
    ]

    # top-level metrics
    metrics["confusion_matrix"] = cm
    metrics["roc"] = roc

    # categorical/numerical columns for drift
    categorical_features = [
        f
        for f in list(data.select_dtypes(include=["category", "object"]))
        if f in df_baseline.columns
    ]
    numerical_features = [
        f for f in df_baseline.columns if f not in categorical_features
    ]
    numerical_features = [
        x
        for x in numerical_features
        if x not in ["id", "score", "label_value", "predicted_probs"]
    ]

    # assigning metrics to output dictionary
    metrics["bias"] = [get_bias_metrics(data)]
    metrics["data_drift"] = get_data_drift_metrics(
        df_baseline, data, numerical_features, categorical_features
    )
    metrics["concept_drift"] = get_concept_drift_metrics(df_baseline, data)
    metrics["interpretability"] = [get_shap_values(data_processed)]

    # MOC expects the action function to be a *yield* function
    yield metrics
