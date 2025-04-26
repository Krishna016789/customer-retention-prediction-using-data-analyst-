import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, render_template, redirect, url_for, send_file
import numpy as np
import xgboost as xgb
import random
import pickle
import pandas as pd
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
import shap
import matplotlib.pyplot as plt
import io
import base64
import time  # For timing (optional profiling)
from sklearn.utils import resample


# Load models
try:
    models = pickle.load(open('xgboosts_model.pkl', 'rb'))  # XGBoost model
    dt_model = pickle.load(open('decision_tree_model.pkl', 'rb'))  # Decision Tree model
    rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))  # Random Forest model
    try:
        voting_model = pickle.load(open('voting_classifier_model.pkl', 'rb'))  # Attempt to load Voting Classifier
        print("Voting Classifier model loaded successfully.")
    except FileNotFoundError:
        voting_model = None  # Fallback to simulation if Voting Classifier file not found
        print("Voting Classifier model file not found. Using simulation (averaging XGBoost, Decision Tree, and Random Forest).")
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    raise  # Re-raise the exception to stop execution if core models fail to load

# Load model accuracies
def load_model_accuracies():
    with open("model_accuracies.pkl", "rb") as f:
        accuracies = pickle.load(f)
    return accuracies

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analysis')
def analysis():
    return render_template("churn_report.html")
@app.route('/analysis2')
def analysis2():
    return render_template("churn_reporttest.html")


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        selected_model = request.form['model']
        threshold = float(request.form.get('threshold', 50))  # Default threshold 50%
        age=int(request.form['age'])
        last_login=int(request.form['last_login'])
        avg_time_spent=float(request.form['avg_time_spent'])
        avg_transaction_value=float(request.form['avg_transaction_value'])
        points_in_wallet=float(request.form['points_in_wallet'])
        date=request.form['date']
        time=request.form['time']
        gender=request.form['gender']
        region_category=request.form['region_category']
        membership_category=request.form['membership_category']
        joined_through_referral=request.form['joined_through_referral']
        preferred_offer_types=request.form['preferred_offer_types']
        medium_of_operation=request.form['medium_of_operation']
        internet_option=request.form['internet_option']
        used_special_discount=request.form['used_special_discount']
        offer_application_preference=request.form['offer_application_preference']
        past_complaint=request.form['past_complaint']
        feedback=request.form['feedback']

        # gender
        if gender == "M":
            gender_M = 1
            gender_F = 0
            gender_Unknown = 0
        elif gender == "F":
            gender_M = 0
            gender_F = 1
            gender_Unknown = 0
        else:  # "Unknown"
         gender_M = 0
         gender_F = 0
         gender_Unknown = 1
        
        # region_category (FIXED)
        if region_category == 'Town':
            region_category_Town = 1
            region_category_Village = 0
            region_category_City = 0
        elif region_category == 'Village':
            region_category_Town = 0
            region_category_Village = 1
            region_category_City = 0
        elif region_category == 'City':
            region_category_Town = 0
            region_category_Village = 0
            region_category_City = 1
        else:
            region_category_Town = 0
            region_category_Village = 0

        # membership_category
        if membership_category=='Gold Membership':
            membership_category_Gold = 1
            membership_category_No = 0
            membership_category_Platinum = 0
            membership_category_Silver = 0
            membership_category_Premium = 0
            membership_category_Basic = 0
        elif membership_category=='No Membership':
            membership_category_Gold = 0
            membership_category_No = 1
            membership_category_Platinum = 0
            membership_category_Silver = 0
            membership_category_Premium = 0
            membership_category_Basic = 0
        elif membership_category=='Platinum Membership':
            membership_category_Gold = 0
            membership_category_No = 0
            membership_category_Platinum = 1
            membership_category_Silver = 0
            membership_category_Premium = 0
            membership_category_Basic = 0
        elif membership_category=='Silver Membership':
            membership_category_Gold = 0
            membership_category_No = 0
            membership_category_Platinum = 0
            membership_category_Silver = 1
            membership_category_Premium = 0
            membership_category_Basic = 0
        elif membership_category=='Premium Membership':
            membership_category_Gold = 0
            membership_category_No = 0
            membership_category_Platinum = 0
            membership_category_Silver = 0
            membership_category_Premium = 1
            membership_category_Basic = 0

        elif membership_category == 'Basic Membership':
            membership_category_Gold = 0
            membership_category_No = 0
            membership_category_Platinum = 0
            membership_category_Silver = 0
            membership_category_Premium = 0
            membership_category_Basic = 1    
        else:
            membership_category_Gold = 0
            membership_category_No = 0
            membership_category_Platinum = 0
            membership_category_Silver = 0
            membership_category_Premium = 0

        # joined_through_referral
        if joined_through_referral=='No':
            joined_through_referral_No = 1
            joined_through_referral_Yes = 0
        elif joined_through_referral=='Yes':
            joined_through_referral_No = 0
            joined_through_referral_Yes = 1
        else:
            joined_through_referral_No = 0
            joined_through_referral_Yes = 0

        # preferred_offer_types
        if preferred_offer_types=='Gift Vouchers/Coupons':
            preferred_offer_types_Gift_VouchersCoupons=1
            preferred_offer_types_Without_Offers=0
            preferred_offer_types_Credit_Debit_Card_Offers = 0
        elif preferred_offer_types == 'Credit/Debit Card Offers':  # Added this condition
            preferred_offer_types_Gift_VouchersCoupons = 0
            preferred_offer_types_Without_Offers = 0
            preferred_offer_types_Credit_Debit_Card_Offers = 1  # Set flag for this offer type    
        elif preferred_offer_types=='Without Offers':
            preferred_offer_types_Gift_VouchersCoupons=0
            preferred_offer_types_Without_Offers=1
            preferred_offer_types_Credit_Debit_Card_Offers = 0
        else:
            preferred_offer_types_Gift_VouchersCoupons=0
            preferred_offer_types_Without_Offers=0

        # medium_of_operation
        if medium_of_operation=='Desktop':
            medium_of_operation_Desktop = 1
            medium_of_operation_Both=0
            medium_of_operation_Smartphone=0
        elif medium_of_operation=='Both':
            medium_of_operation_Desktop = 0
            medium_of_operation_Both=1
            medium_of_operation_Smartphone=0
        elif medium_of_operation=='Smartphone':
            medium_of_operation_Desktop = 0
            medium_of_operation_Both=0
            medium_of_operation_Smartphone=1
        else:
            medium_of_operation_Desktop = 0
            medium_of_operation_Both=0
            medium_of_operation_Smartphone=0

    # internet_option
        if internet_option == 'Mobile_Data':
            internet_option_Mobile_Data = 1
            internet_option_Wi_Fi=0
            internet_option_Fiber_Optic = 0
        elif internet_option == 'Wi-Fi':
            internet_option_Mobile_Data = 0
            internet_option_Wi_Fi=1
            internet_option_Fiber_Optic = 0
        elif internet_option == 'Fiber_Optic':
            internet_option_Mobile_Data = 0
            internet_option_Wi_Fi = 0
            internet_option_Fiber_Optic = 1    
        else:
            internet_option_Mobile_Data = 0
            internet_option_Wi_Fi=0

        # used_special_discount (FIXED)
        if used_special_discount == 'Yes':
            used_special_discount_Yes = 1
        else:
            used_special_discount_Yes = 0

        # offer_application_preference (FIXED)
        if offer_application_preference == 'Yes':
            offer_application_preference_Yes = 1
        else:
            offer_application_preference_Yes = 0

        # past_complaint (FIXED)
        if past_complaint == 'Yes':
            past_complaint_Yes = 1
        else:
            past_complaint_Yes = 0

        # feedback
        if feedback =='Poor Customer Service':
            feedback_Customer=1
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
            feedback_No_Reason_Specified = 0
        elif feedback =='Poor Product Quality':
            feedback_Customer=0
            feedback_Poor_Product_Quality=1
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
            feedback_No_Reason_Specified = 0
        elif feedback =='Poor Website':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=1
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
            feedback_No_Reason_Specified = 0
        elif feedback =='Products always in Stock':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=1
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
            feedback_No_Reason_Specified = 0
        elif feedback =='Quality Customer Care':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=1
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
            feedback_No_Reason_Specified = 0
        elif feedback =='Reasonable Price':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=1
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0
            feedback_No_Reason_Specified = 0
        elif feedback =='Too many ads':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=1
            feedback_User_Friendly_Website=0
            feedback_No_Reason_Specified = 0
        elif feedback =='User Friendly Website':
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=1
            feedback_No_Reason_Specified = 0
        elif feedback == 'No reason specified':
             feedback_Customer = 0
             feedback_Poor_Product_Quality = 0
             feedback_Poor_Website = 0
             feedback_Products_always_in_Stock = 0
             feedback_Quality_Customer_Care = 0
             feedback_Reasonable_Price = 0
             feedback_Too_many_ads = 0
             feedback_User_Friendly_Website = 0
             feedback_No_Reason_Specified = 1 
        else:
            feedback_Customer=0
            feedback_Poor_Product_Quality=0
            feedback_Poor_Website=0
            feedback_Products_always_in_Stock=0
            feedback_Quality_Customer_Care=0
            feedback_Reasonable_Price=0
            feedback_Too_many_ads=0
            feedback_User_Friendly_Website=0

        date2 = date.split('-')
        joining_day=int(date2[0])
        joining_month=int(date2[1])
        joining_year=int(date2[2])

        time2 = time.split(':')
        last_visit_time_hour=int(time2[0])
        last_visit_time_minutes=int(time2[1])
        last_visit_time_seconds=int(time2[2])

        data = {'age':[age], 'days_since_last_login':[last_login], 'avg_time_spent':[avg_time_spent], 'avg_transaction_value':[avg_transaction_value], 'points_in_wallet':[points_in_wallet], 'joining_day':[joining_day], 'joining_month':[joining_month], 'joining_year':[joining_year], 'last_visit_time_hour':[last_visit_time_hour], 'last_visit_time_minutes':[last_visit_time_minutes], 'last_visit_time_seconds':[last_visit_time_seconds], 'gender_M':[gender_M],'gender_F': [gender_F], 'gender_Unknown':[gender_Unknown], 'region_category_Town':[region_category_Town],'region_category_City': [region_category_City], 'region_category_Village':[region_category_Village], 'membership_category_Gold Membership':[membership_category_Gold], 'membership_category_No Membership':[membership_category_No],'membership_category_Basic Membership': [membership_category_Basic], 'membership_category_Platinum Membership':[membership_category_Platinum], 'membership_category_Premium Membership':[membership_category_Premium], 'membership_category_Silver Membership':[membership_category_Silver], 'joined_through_referral_No':[joined_through_referral_No], 'joined_through_referral_Yes':[joined_through_referral_Yes], 'preferred_offer_types_Gift Vouchers/Coupons':[preferred_offer_types_Gift_VouchersCoupons], 'preferred_offer_types_Without Offers':[preferred_offer_types_Without_Offers],'preferred_offer_types_Credit_Debit_Card_Offers':[preferred_offer_types_Credit_Debit_Card_Offers], 'medium_of_operation_Both':[medium_of_operation_Both], 'medium_of_operation_Desktop':[medium_of_operation_Desktop], 'medium_of_operation_Smartphone':[medium_of_operation_Smartphone], 'internet_option_Mobile_Data':[internet_option_Mobile_Data], 'internet_option_Wi-Fi':[internet_option_Wi_Fi], 'internet_option_Fiber_Optic': [internet_option_Fiber_Optic], 'used_special_discount_Yes':[used_special_discount_Yes], 'offer_application_preference_Yes':[offer_application_preference_Yes], 'past_complaint_Yes':[past_complaint_Yes], 'feedback_Poor Customer Service':[feedback_Customer], 'feedback_Poor Product Quality':[feedback_Poor_Product_Quality], 'feedback_Poor Website':[feedback_Poor_Website],  'feedback_Products always in Stock':[feedback_Products_always_in_Stock], 'feedback_Quality Customer Care':[feedback_Quality_Customer_Care], 'feedback_Reasonable Price':[feedback_Reasonable_Price], 'feedback_Too many ads':[feedback_Too_many_ads], 'feedback_User Friendly Website':[feedback_User_Friendly_Website],'feedback_No reason specified': [feedback_No_Reason_Specified]}

        import pandas as pd
        df = pd.DataFrame.from_dict(data)

        cols = models.get_booster().feature_names
        df = df[cols]

    # --- CHANGE 1: Model Prediction with Explainability ---
        if selected_model == "XGBoost":
            raw_score = models.predict(df)[0]
            score = raw_score * 20  # **HIGHLIGHT**: Scale score from 0-5 to 0-100
            probability = models.predict_proba(df)[:, 1][0] * 100 if hasattr(models, "predict_proba") else 100 / (1 + np.exp(-(raw_score - 2)))
            explainer = shap.TreeExplainer(models)
            feature_importance = models.get_booster().get_score(importance_type='gain')
        elif selected_model == "Decision Tree":
            raw_score = dt_model.predict(df)[0]
            score = raw_score * 20  # **HIGHLIGHT**: Scale score from 0-5 to 0-100
            probability = dt_model.predict_proba(df)[:, 1][0] * 100 if hasattr(dt_model, "predict_proba") else 100 / (1 + np.exp(-(raw_score - 2)))
            explainer = shap.TreeExplainer(dt_model)
            feature_importance = dict(zip(df.columns, dt_model.feature_importances_))
        elif selected_model == "Random Forest":
            raw_score = rf_model.predict(df)[0]
            score = raw_score * 20  # **HIGHLIGHT**: Scale score from 0-5 to 0-100
            probability = rf_model.predict_proba(df)[:, 1][0] * 100 if hasattr(rf_model, "predict_proba") else 100 / (1 + np.exp(-(raw_score - 2)))
            explainer = shap.TreeExplainer(rf_model)
            feature_importance = dict(zip(df.columns, rf_model.feature_importances_))
        elif selected_model == "Voting Classifier":
            xgb_score = models.predict(df)[0]
            dt_score = dt_model.predict(df)[0]
            rf_score = rf_model.predict(df)[0]
            raw_score = np.mean([xgb_score, dt_score, rf_score])
            score = raw_score * 20  # **HIGHLIGHT**: Scale score from 0-5 to 0-100
            if hasattr(models, "predict_proba") and hasattr(dt_model, "predict_proba") and hasattr(rf_model, "predict_proba"):
                xgb_prob = models.predict_proba(df)[:, 1][0] * 100
                dt_prob = dt_model.predict_proba(df)[:, 1][0] * 100
                rf_prob = rf_model.predict_proba(df)[:, 1][0] * 100
                probability = np.mean([xgb_prob, dt_prob, rf_prob])
            else:
                probability = 100 / (1 + np.exp(-(raw_score - 2)))
            explainer = shap.TreeExplainer(rf_model)  # Use RF as proxy
            feature_importance = dict(zip(df.columns, rf_model.feature_importances_))
        else:
            score = 0
            probability = 0
            
        # SHAP Explanations - Fixed for single prediction
        shap_values = explainer.shap_values(df)
        # Ensure we use the first (and only) sample's SHAP values
        # For binary classification, shap_values may have shape (n_samples, n_features, 2); we want the positive class (index 1)
        if len(shap_values.shape) == 3:  # Multi-output case
            shap_values_single = shap_values[0, :, 1]  # First sample, positive class
            base_value = explainer.expected_value[1]  # Positive class base value
        else:  # Single-output case
            shap_values_single = shap_values[0]  # First sample
            base_value = explainer.expected_value
        # Use Agg backend to avoid GUI threading issues in Flask
        import matplotlib
        matplotlib.use('Agg')
        plt.figure(figsize=(10, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_single,
                base_values=base_value,
                data=df.iloc[0],
                feature_names=df.columns.tolist()
            ),
            show=False
        )
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        shap_waterfall_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Feature Importance Chart
        feature_names = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importance_values, color='skyblue')
        plt.xlabel('Importance')
        plt.title(f'{selected_model} Feature Importance')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        feature_importance_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        # --- END CHANGES 1-3 ---


        probability = min(max(probability, 0), 100)
        churn_status = "Churn" if probability > threshold else "No Churn"

        # Ensure probability is between 0-100%
        probability = min(max(probability, 0), 100)

        # Calculate churn percentage based on score
        if score < 30:
            percentage = 20  # Low risk
        elif score < 60:
            percentage = 50  # Medium risk
        else:
            percentage = 80  # High risk

        print(f"DEBUG: Model={selected_model}, Score={score}, Probability={probability}, Percentage={percentage}")

        # Prepare df_results to pass to graph.html
        df_results = [{
            'customer_index': f"1 ({churn_status})",
            'age': int(age),  # Convert to native int
            'days_since_last_login': int(last_login),  # Convert to native int
            'avg_time_spent': float(avg_time_spent),  # Convert to native float
            'avg_transaction_value': float(avg_transaction_value),  # Convert to native float
            'points_in_wallet': float(points_in_wallet),  # Convert to native float
            'score': float(score),  # Convert to native float
            'probability': float(probability),  # Convert to native float
            'percentage': float(percentage)  # Convert to native float
        }]
        # Serialize df_results to JSON in Python
        df_results_json = json.dumps(df_results)

        # Return the prediction page (removed redirect to graph for simplicity)
        return render_template(
            "prediction.html",
            prediction_text=f"{selected_model} Churn Score is {score:.2f}%",
            prediction_probability=f"{selected_model} Churn Probability is {probability:.2f}%",
            prediction_percentage=f"{selected_model} Churn Percentage is {percentage:.2f}%",
            selected_model=selected_model,
            score=score,
            probability=probability,
            percentage=percentage,
            threshold=threshold,
            df_results_json=df_results_json,
            df_results=df_results,
            shap_waterfall_img=shap_waterfall_img,  # New: SHAP waterfall plot
            feature_importance_img=feature_importance_img  # New: Feature importance chart
        )
    else:
        return render_template("prediction.html")

@app.route('/graph')
def graph():
    # Existing parameters
    score = float(request.args.get('score', 0))
    probability = float(request.args.get('probability', 0))
    percentage = float(request.args.get('percentage', 0))
    selected_model = request.args.get('model', 'Unknown')
    age = int(request.args.get('age', 0))
    last_login = int(request.args.get('days_since_last_login', 0))
    avg_time_spent = float(request.args.get('avg_time_spent', 0))
    avg_transaction_value = float(request.args.get('avg_transaction_value', 0))
    points_in_wallet = float(request.args.get('points_in_wallet', 0))

    # Prepare df_results with all required fields
    df_results = [{
        'customer_index': f"1 ({'Churn' if probability > 50 else 'No Churn'})",
        'age': age,
        'days_since_last_login': last_login,
        'avg_time_spent': avg_time_spent,
        'avg_transaction_value': avg_transaction_value,
        'points_in_wallet': points_in_wallet,
        'score': score,
        'probability': probability,
        'percentage': percentage,
        'membership_category': 'No Membership',  # Default value
        'feedback': 'No reason specified',      # Default value
        'region': 'City',                      # Default value
        'referral': 'No'                       # Default value
    }]

    # Simulate additional customers for dashboard visualizations
    np.random.seed(42)  # For reproducibility
    n_customers = 50
    simulated_data = [{
        'customer_index': f"{i} ({'Churn' if probability_value > 50 else 'No Churn'})",
        'age': int(np.random.randint(18, 80)),
        'days_since_last_login': int(np.random.randint(1, 60)),
        'avg_time_spent': float(np.random.uniform(10, 500)),
        'avg_transaction_value': float(np.random.uniform(50, 1000)),
        'points_in_wallet': float(np.random.uniform(0, 2000)),
        'score': float(np.random.uniform(0, 100)),
        'probability': probability_value,
        'percentage': float(np.random.uniform(0, 100)),
        'membership_category': np.random.choice(['Basic', 'Gold', 'Platinum', 'No Membership', 'Silver', 'Premium']),
        'feedback': np.random.choice(['Poor Customer Service', 'Reasonable Price', 'Poor Website', 'No reason specified', 'Quality Customer Care']),
        'region': np.random.choice(['Town', 'City', 'Village']),
        'referral': np.random.choice(['Yes', 'No'])
    } for i, probability_value in enumerate(np.random.uniform(0, 100, n_customers), start=2)]

    # Include the original prediction
    simulated_data.insert(0, df_results[0])

    # Prepare dashboard chart data
    dashboard_chart_data = {
        # Earlier Set
        'heatmap': {
            'z': [[np.mean([d['probability'] for d in simulated_data if d['region'] == r and d['membership_category'] == m] or [0])
                   for m in ['Basic', 'Gold', 'Platinum', 'No Membership', 'Silver', 'Premium']]
                  for r in ['Town', 'City', 'Village']],
            'x': ['Basic', 'Gold', 'Platinum', 'No Membership', 'Silver', 'Premium'],
            'y': ['Town', 'City', 'Village']
        },
        'histogram': {
            'probabilities': [d['probability'] for d in simulated_data]
        },
        'table_data': simulated_data[:10],  # Top 10 customers
        'timeseries': {
            'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            'probabilities': [np.mean([d['probability'] for d in simulated_data]) * np.random.uniform(0.8, 1.2) for _ in range(5)]
        },
        'sankey': {
            'labels': ['All Customers', 'Active', 'Engaged', 'Retained', 'Churned'],
            'source': [0, 0, 1, 1, 2],
            'target': [1, 2, 2, 3, 4],
            'value': [len(simulated_data),
                      sum(1 for d in simulated_data if d['days_since_last_login'] < 30),
                      sum(1 for d in simulated_data if d['days_since_last_login'] >= 30),
                      sum(1 for d in simulated_data if d['probability'] < 50),
                      sum(1 for d in simulated_data if d['probability'] >= 50)]
        },
        # Second Set
        'treemap': {
            'labels': [d['membership_category'] for d in simulated_data],
            'values': [d['probability'] for d in simulated_data],
            'parents': [''] * len(simulated_data)
        },
        'radar': {
            'churned': [np.mean([d[k] for d in simulated_data if d['probability'] > 50] or [0])
                        for k in ['age', 'days_since_last_login', 'avg_time_spent', 'avg_transaction_value', 'points_in_wallet']],
            'retained': [np.mean([d[k] for d in simulated_data if d['probability'] <= 50] or [0])
                         for k in ['age', 'days_since_last_login', 'avg_time_spent', 'avg_transaction_value', 'points_in_wallet']],
            'categories': ['Age', 'Days Since Login', 'Time Spent', 'Transaction Value', 'Points']
        },
        'wordcloud': {
            'words': [d['feedback'] for d in simulated_data],
            'counts': pd.Series([d['feedback'] for d in simulated_data]).value_counts().to_dict()
        },
        'funnel': {
            'stages': ['Total', 'Active', 'Engaged', 'Retained'],
            'values': [len(simulated_data),
                       sum(1 for d in simulated_data if d['days_since_last_login'] < 30),
                       sum(1 for d in simulated_data if d['avg_time_spent'] > 100),
                       sum(1 for d in simulated_data if d['probability'] < 50)]
        },
        'bubble': {
            'x': [d['age'] for d in simulated_data],
            'y': [d['probability'] for d in simulated_data],
            'sizes': [d['points_in_wallet'] / 20 for d in simulated_data]
        },
        # Newly Proposed
        'parallel': {
            'data': [[d['age'], d['days_since_last_login'], d['avg_time_spent'], d['avg_transaction_value'], d['probability']]
                     for d in simulated_data],
            'dimensions': ['Age', 'Days Since Login', 'Time Spent', 'Transaction Value', 'Probability']
        },
        'violin': {
            'categories': list(set(d['membership_category'] for d in simulated_data)),
            'probabilities': [[d['probability'] for d in simulated_data if d['membership_category'] == c]
                             for c in set(d['membership_category'] for d in simulated_data)]
        },
        'area': {
            'high_risk': [sum(1 for d in simulated_data if d['probability'] > 75) * np.random.uniform(0.8, 1.2) for _ in range(5)],
            'medium_risk': [sum(1 for d in simulated_data if 25 <= d['probability'] <= 75) * np.random.uniform(0.8, 1.2) for _ in range(5)],
            'low_risk': [sum(1 for d in simulated_data if d['probability'] < 25) * np.random.uniform(0.8, 1.2) for _ in range(5)],
            'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May']
        },
        'hexbin': {
            'x': [d['age'] for d in simulated_data],
            'y': [d['probability'] for d in simulated_data]
        },
        'network': {
            'nodes': [f"Customer {d['customer_index'].split()[0]}" for d in simulated_data],
            'edges': [(i, j) for i in range(len(simulated_data)) for j in range(i + 1, len(simulated_data))
                      if simulated_data[i]['referral'] == 'Yes' and simulated_data[j]['referral'] == 'Yes' and np.random.random() > 0.8][:10],
            'sizes': [d['probability'] for d in simulated_data]
        }
    }

    # Rest of the existing code
    churn_count = sum(1 for result in df_results if result['probability'] > 50)
    accuracies = load_model_accuracies()
    insights = [
        f"Predicted {churn_count} customer(s) to churn (Probability > 50%).",
        f"Churn Probability: {probability:.2f}%"
    ]
    recommendations = []
    if churn_count > 0:
        recommendations.append(f"Focus retention efforts on {churn_count} high-risk customer(s).")
    else:
        recommendations.append("Maintain current engagement strategies.")
    recommendations.append(f"Monitor customer with score: {score:.2f}.")

    chart_data = {
        'scores': [r['score'] for r in df_results],
        'probabilities': [r['probability'] for r in df_results],
        'percentages': [r['percentage'] for r in df_results],
        'labels': [r['customer_index'] for r in df_results],
        'risk_distribution': [
            sum(1 for r in df_results if r['percentage'] < 30),
            sum(1 for r in df_results if 30 <= r['percentage'] < 60),
            sum(1 for r in df_results if r['percentage'] >= 60)
        ],
        'ages': [r['age'] for r in df_results],
        'days_since_last_login': [r['days_since_last_login'] for r in df_results],
        'avg_time_spent': [r['avg_time_spent'] for r in df_results]
    }
    insights_chart_data = {
        'probability_vs_age': {'x': chart_data['ages'], 'y': chart_data['probabilities']},
        'score_distribution': chart_data['scores'],
        'time_spent_vs_prob': {'x': chart_data['avg_time_spent'], 'y': chart_data['probabilities']},
        'login_vs_prob': {'x': chart_data['days_since_last_login'], 'y': chart_data['probabilities']},
        'risk_pie': chart_data['risk_distribution'],
        'score_gauge': chart_data['scores'][0]
    }
    recommendations_chart_data = {
        'churn_bar': chart_data['scores'],
        'prob_trend': chart_data['probabilities'],
        'percentage_pie': chart_data['risk_distribution'],
        'age_vs_score': {'x': chart_data['ages'], 'y': chart_data['scores']},
        'time_spent_bar': chart_data['avg_time_spent'],
        'risk_gauge': chart_data['probabilities'][0]
    }

    return render_template(
        'graph.html',
        df_results=df_results,
        insights=insights,
        recommendations=recommendations,
        chart_data=json.dumps(chart_data),
        insights_chart_data=json.dumps(insights_chart_data),
        recommendations_chart_data=json.dumps(recommendations_chart_data),
        dashboard_chart_data=json.dumps(dashboard_chart_data),
        selected_model=selected_model,
        rf_accuracy=accuracies.get("Random Forest", "N/A"),
        dt_accuracy=accuracies.get("Decision Tree", "N/A"),
        xgb_accuracy=accuracies.get("XGBoost", "N/A"),
        ensemble_accuracy=accuracies.get("Ensemble", "N/A")
    )

@app.route('/batch_prediction', methods=['GET', 'POST'])
def batch_prediction():
    if request.method == "POST":
        # Check if file is present in the request
        if 'file' not in request.files:
            return render_template("batch_prediction.html", error="No file uploaded")
        file = request.files['file']
        selected_model = request.form.get('model', 'XGBoost')  # Default to XGBoost
        year = request.form.get('year', type=int)  # Get year as integer
        month = request.form.get('month', type=int)  # Get month as integer (None if empty)
        custom_model_file = request.files.get('custom_model')  # Get custom model file
        
        # --- CHANGE 1: Updated file validation to support Excel (.xlsx) ---
        # Validate file
        if file.filename == '':
            return render_template("batch_prediction.html", error="No file selected")
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):  # Updated to accept .xlsx
            return render_template("batch_prediction.html", error="File must be CSV or Excel (.xlsx)")
        if selected_model == "Custom Model" and (not custom_model_file or not custom_model_file.filename.endswith('.pkl')):
            return render_template("batch_prediction.html", error="Please upload a valid .pkl file for Custom Model")
        # --- END CHANGE 1 ---
   
        try:
            start_time = time.time()  # Optional: Profile execution time
            # --- CHANGE 2: Added Excel file reading support ---
            # Read file based on extension
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:  # .xlsx
                df = pd.read_excel(file)  # New support for Excel files
            # --- END CHANGE 2 ---
            
            required_cols = models.get_booster().feature_names
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return render_template("batch_prediction.html", 
                                    error=f"Missing required columns: {', '.join(missing_cols)}")
            
            # --- New: Filter or Adjust Data Based on Year/Month ---
            # Assuming dataset has 'joining_year' and 'joining_month' columns
           # --- Changes: Enhanced Year/Month Filtering ---
            filtered_df = df.copy()  # Work with a copy to preserve original data
            if 'joining_year' in df.columns and 'joining_month' in df.columns:
                if year and month:
                    filtered_df = df[(df['joining_year'] == year) & (df['joining_month'] == month)]
                    if filtered_df.empty:
                        return render_template("batch_prediction.html", 
                                            error=f"No data found for Year: {year}, Month: {month}. Proceeding with full dataset.",
                                            year=year, month=month, use_full_dataset=True)
                elif year:
                    filtered_df = df[df['joining_year'] == year]
                    if filtered_df.empty:
                        return render_template("batch_prediction.html", 
                                            error=f"No data found for Year: {year}. Proceeding with full dataset.",
                                            year=year, month=month, use_full_dataset=True)
                df = filtered_df if not filtered_df.empty else df  # Use filtered data if available, else full dataset
            else:
                # If no year/month columns, warn but proceed with full dataset
                print("Warning: Dataset lacks 'joining_year' or 'joining_month' columns. Using full dataset.")
                if year or month:
                    return render_template("batch_prediction.html", 
                                        error="Dataset lacks 'joining_year' or 'joining_month' columns. Proceeding with full dataset.",
                                        year=year, month=month, use_full_dataset=True)

            
            # --- CHANGE 3: Enhanced data validation and cleaning ---
            # Data Validation and Cleaning
            original_shape = df.shape
            validation_report = {
                'total_rows': original_shape[0],
                'total_columns': original_shape[1],
                'missing_values_before': df.isnull().sum().to_dict(),  # New: Track missing values before cleaning
                'data_types': df.dtypes.astype(str).to_dict(),
                'summary_stats': df.describe().to_dict()
            }
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            # Auto-fill numeric columns with median
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val)
            
            # Auto-fill categorical columns with mode or 'Unknown'
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val)
            
            # Suggest corrections for out-of-range numeric values
            corrections = {}
            for col in numeric_cols:
                min_val, max_val = df[col].min(), df[col].max()
                if col in ['age', 'days_since_last_login', 'avg_time_spent', 
                         'avg_transaction_value', 'points_in_wallet']:
                    if min_val < 0:  # Negative values are invalid
                        corrections[col] = f"Negative values detected (min: {min_val}). Suggest clipping to 0."
                        df[col] = df[col].clip(lower=0)
                    if col == 'age' and max_val > 120:  # Unrealistic age
                        corrections[col] = f"Age > 120 detected (max: {max_val}). Suggest capping at 120."
                        df[col] = df[col].clip(upper=120)
            
            # Update validation report after cleaning
            validation_report['missing_values_after'] = df.isnull().sum().to_dict()  # New: Track after cleaning
            validation_report['corrections_suggested'] = corrections  # New: Add correction suggestions
            # --- END CHANGE 3 ---
            
            df = df[required_cols]

            # --- CHANGE 1: Model Prediction with Explainability ---
            
           # HIGHLIGHT: Model Prediction and Comparison Logic
            chunk_size = 1000
            model_list = {
                "XGBoost": models,
                "Decision Tree": dt_model,
                "Random Forest": rf_model,
                "Voting Classifier": None
            }
            if selected_model == "Custom Model" and custom_model_file:
                custom_model = pickle.load(custom_model_file)
                model_list["Custom Model"] = custom_model

            model_comparison = {}
            predictions = []
            probabilities = []
            for model_name, model in model_list.items():
                model_start_time = time.time()
                model_preds = []
                model_probs = []
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i + chunk_size]
                    if model_name == "Voting Classifier":
                        xgb_preds = models.predict(chunk)
                        dt_preds = dt_model.predict(chunk)
                        rf_preds = rf_model.predict(chunk)
                        pred = np.mean([xgb_preds, dt_preds, rf_preds], axis=0)
                        if all(hasattr(m, "predict_proba") for m in [models, dt_model, rf_model]):
                            xgb_probs = models.predict_proba(chunk)[:, 1] * 100
                            dt_probs = dt_model.predict_proba(chunk)[:, 1] * 100
                            rf_probs = rf_model.predict_proba(chunk)[:, 1] * 100
                            prob = np.mean([xgb_probs, dt_probs, rf_probs], axis=0)
                        else:
                            prob = [100 / (1 + np.exp(-(p / 20 - 2))) for p in pred]
                    else:
                        pred = model.predict(chunk)
                        prob = model.predict_proba(chunk)[:, 1] * 100 if hasattr(model, "predict_proba") else [random.uniform(10, 95) for _ in pred]
                    model_preds.extend(pred)
                    model_probs.extend(prob)
                
                model_time = time.time() - model_start_time
                avg_churn_score = np.mean(model_preds)
                avg_probability = np.mean(model_probs)
                model_comparison[model_name] = {
                    "avg_churn_score": avg_churn_score,
                    "avg_probability": avg_probability,
                    "prediction_time": model_time
                }
                if model_name == selected_model:
                    predictions = model_preds
                    probabilities = model_probs
                    explainer = shap.TreeExplainer(rf_model if model_name == "Voting Classifier" else model)
                    feature_importance = (model.get_booster().get_score(importance_type='gain') if model_name == "XGBoost" else
                     dict(zip(df.columns, model.feature_importances_)) if model_name in ["Decision Tree", "Random Forest"] else
                     dict(zip(df.columns, rf_model.feature_importances_)) if model_name == "Voting Classifier" else
                     dict(zip(df.columns, model.feature_importances_)))

            max_shap_samples = 100
            if len(df) > max_shap_samples:
                df_sample = resample(df, n_samples=max_shap_samples, random_state=42)
            else:
                df_sample = df

            shap_values = explainer.shap_values(df_sample)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, df_sample, show=False, max_display=15, plot_type="bar", feature_names=df_sample.columns.tolist())
            plt.title(f'SHAP Summary Plot - Top 15 Features ({selected_model})', fontsize=12)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            shap_summary_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
 
            feature_names = list(feature_importance.keys())
            importance_values = list(feature_importance.values())
            sorted_indices = np.argsort(importance_values)[::-1][:15]
            top_feature_names = [feature_names[i] for i in sorted_indices]
            top_importance_values = [importance_values[i] for i in sorted_indices]
            plt.figure(figsize=(10, 6))
            plt.barh(top_feature_names, top_importance_values, color='skyblue')
            plt.xlabel('Importance Score', fontsize=10)
            plt.ylabel('Features', fontsize=10)
            plt.title(f'Top 15 Feature Importance ({selected_model})', fontsize=12)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            feature_importance_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            results = [(i + 1, pred, prob) for i, (pred, prob) in enumerate(zip(predictions, probabilities))]
            page = request.args.get('page', 1, type=int)
            per_page = 100
            total_records = len(results)
            total_pages = (total_records + per_page - 1) // per_page
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, total_records)
            paginated_results = results[start_idx:end_idx]

            avg_churn_score = np.mean([pred for _, pred, _ in results])
            avg_probability = np.mean([prob for _, _, prob in results])
            churn_count = sum(1 for _, _, prob in results if prob > 50)
            total_count = len(results)
            churn_percentage = (churn_count / total_count) * 100 if total_count > 0 else 0

            high_risk_count = sum(1 for _, _, prob in results if prob > 75)
            low_risk_count = sum(1 for _, _, prob in results if prob < 25)
            medium_risk_count = total_count - high_risk_count - low_risk_count
            score_distribution = [sum(1 for _, pred, _ in results if pred < 30), 
                                sum(1 for _, pred, _ in results if 30 <= pred < 60), 
                                sum(1 for _, pred, _ in results if pred >= 60)]

            insights = [
                ("High churn risk detected: Over 50% of customers are likely to leave." if churn_percentage > 50 else 
                 "Moderate churn risk: 20-50% of customers may churn." if churn_percentage > 20 else 
                 "Low churn risk: Less than 20% of customers are at risk.", churn_percentage),
                ("Risk distribution across probability ranges.", [low_risk_count, medium_risk_count, high_risk_count]),
                ("Score distribution across ranges.", score_distribution)
            ]

            recommendations = [
                ("Offer targeted discounts or loyalty rewards to retain high-risk customers.", high_risk_count) if avg_probability > 50 else ("Maintain current strategies.", 0),
                (f"Focus retention efforts on the {churn_count} customers with >50% churn probability.", churn_count) if churn_count > 0 else ("No immediate action needed.", 0),
                ("Monitor low-risk customers to maintain satisfaction.", low_risk_count)
            ]

            prob_histogram = np.histogram([prob for _, _, prob in results], bins=10, range=(0, 100))[0].tolist()
            score_boxplot = {
                'min': float(min([pred for _, pred, _ in results])),
                'q1': float(np.percentile([pred for _, pred, _ in results], 25)),
                'median': float(np.median([pred for _, pred, _ in results])),
                'q3': float(np.percentile([pred for _, pred, _ in results], 75)),
                'max': float(max([pred for _, pred, _ in results]))
            }

            chart_data = {
                "predictions": [float(pred) for _, pred, _ in results],
                "probabilities": [float(prob) for _, _, prob in results],
                "labels": [f"Customer {i}" for i, _, _ in results],
                "churn_distribution": [churn_count, total_count - churn_count],
                "risk_distribution": [low_risk_count, medium_risk_count, high_risk_count],
                "score_distribution": score_distribution,
                "risk_counts": [high_risk_count, medium_risk_count, low_risk_count],
                "gauges": {
                    "avg_churn_score": float(avg_churn_score),
                    "avg_probability": float(avg_probability),
                    "churn_percentage": float(churn_percentage)
                },
                "prob_histogram": prob_histogram,
                "score_boxplot": score_boxplot
            }

            execution_time = time.time() - start_time
            print(f"Batch prediction took {execution_time:.2f} seconds")

            # HIGHLIGHT: Updated render_template with model_comparison
            return render_template(
                "batch_prediction.html",
                results=paginated_results,
                total_records=total_records,
                page=page,
                per_page=per_page,
                total_pages=total_pages,
                avg_churn_score=avg_churn_score,
                avg_probability=avg_probability,
                churn_percentage=churn_percentage,
                insights=insights,
                recommendations=recommendations,
                chart_data=json.dumps(chart_data),
                selected_model=selected_model,
                validation_report=validation_report,
                shap_summary_img=shap_summary_img,
                feature_importance_img=feature_importance_img,
                year=year,
                month=month,
                model_comparison=model_comparison  # New parameter for comparison
            )
        except Exception as e:
            return render_template("batch_prediction.html", error=f"Error processing file: {str(e)}")
    
    return render_template("batch_prediction.html")

# Export Routes
@app.route('/export_pdf/<type>', methods=['GET'])
def export_pdf(type):
    if type == "single":
        df_results = request.args.get('df_results')
        df_results = json.loads(df_results)
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = [Paragraph("Single Churn Prediction Report", styles['Title']), Spacer(1, 12)]
        data = [["Customer", "Age", "Last Login", "Time Spent", "Transaction", "Points", "Score", "Probability", "Percentage"]] + [[r['customer_index'], r['age'], r['days_since_last_login'], r['avg_time_spent'], r['avg_transaction_value'], r['points_in_wallet'], r['score'], r['probability'], r['percentage']] for r in df_results]
        t = Table(data)
        elements.append(t)
        doc.build(elements)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="single_prediction.pdf", mimetype='application/pdf')
    
@app.route('/export_excel/<type>', methods=['GET'])
def export_excel(type):
    if type == "single":
        df_results = json.loads(request.args.get('df_results'))
        df = pd.DataFrame(df_results)
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="single_prediction.xlsx", mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
   
if __name__ == "__main__":
    app.run(debug=True)

