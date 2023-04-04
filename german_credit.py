# modelop.schema.0: schema_input.avsc
# modelop.schema.1: schema_output.avsc

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
def metrics(data):
    
    yield {"foo":"bar"}
