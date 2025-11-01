from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained pipeline
clf = joblib.load("customer_behavior_model.pkl")

print("="*50)
print("MODEL LOADED SUCCESSFULLY")
print("="*50)

class CustomerInput(BaseModel):
    age: float
    gender: str
    size: str
    color: str
    season: str
    purchase_amount: float
    review_rating: float
    purchase_frequency_days: float
    discount_applied: str
    promo_code_used: str
    subscription_status: str


def engineer_features(df):

   
    if 'age_group' not in df.columns:
        def categorize_age(age):
            if age <= 25:
                return 'Young Adult'
            elif age <= 40:
                return 'Adult'
            elif age <= 59:
                return 'Middle-Aged'
            else:
                return 'Elderly'
        df['age_group'] = df['age'].apply(categorize_age)
    
    
    if 'frequency_of_purchases' not in df.columns:
        def categorize_frequency(days):
            if days <= 7:
                return 'Weekly'
            elif days <= 14:
                return 'Fortnightly'
            else:
                return 'Monthly'
        df['frequency_of_purchases'] = df['purchase_frequency_days'].apply(categorize_frequency)
    
   
    df['avg_spent_per_day'] = df['purchase_amount'] / (df['purchase_frequency_days'] + 1)
    df['price_per_frequency'] = df['purchase_amount'] / (df['purchase_frequency_days'] + 1)
    
    QUANTILE_25_DAYS = 14  
    df['is_frequent_buyer'] = (df['purchase_frequency_days'] < QUANTILE_25_DAYS).astype(int)
    
   
    df['customer_value'] = df['purchase_amount'] * (30 / (df['purchase_frequency_days'] + 1))
    
    QUANTILE_75_VALUE = 500 
    df['high_value_customer'] = (df['customer_value'] > QUANTILE_75_VALUE).astype(int)
    
    
    df['discount_user'] = df['discount_applied'].map({'Yes': 1, 'No': 0})
    df['promo_user'] = df['promo_code_used'].map({'Yes': 1, 'No': 0})
    df['deal_seeker'] = ((df['discount_user'] == 1) | (df['promo_user'] == 1)).astype(int)
    

    df['is_subscriber'] = df['subscription_status'].map({'Yes': 1, 'No': 0})
    df['loyalty_score'] = df['is_subscriber'] + df['is_frequent_buyer'] + (df['review_rating'] >= 4).astype(int)
    
  
    df['gender_season'] = df['gender'] + '_' + df['season']
    df['age_group_frequency'] = df['age_group'] + '_' + df['frequency_of_purchases']
    
    return df

@app.post("/predict")
def predict(data: CustomerInput):
    try:
        print("\n" + "="*60)
        print("🔍 NEW PREDICTION REQUEST")
        print("="*60)
        
        # Convert to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Show input
        print(f"\n📥 RAW INPUT:")
        print(f"  Purchase Amount: ${input_dict['purchase_amount']}")
        print(f"  Purchase Frequency Days: {input_dict['purchase_frequency_days']}")
        print(f"  Review Rating: {input_dict['review_rating']}")
        print(f"  Subscription: {input_dict['subscription_status']}")
        print(f"  Discount: {input_dict['discount_applied']}")
        
        # Engineer features
        df = engineer_features(df)
        
        # Show engineered features
        print(f"\n⚙️  ENGINEERED FEATURES:")
        print(f"  customer_value: {df['customer_value'].values[0]:.2f}")
        print(f"  high_value_customer: {df['high_value_customer'].values[0]}")
        print(f"  is_frequent_buyer: {df['is_frequent_buyer'].values[0]}")
        print(f"  loyalty_score: {df['loyalty_score'].values[0]}")
        print(f"  deal_seeker: {df['deal_seeker'].values[0]}")
        
        print(f"\n📋 Total features: {len(df.columns)}")
        
        # Make prediction
        y_pred = clf.predict(df)
        predicted_category = int(y_pred[0])
        
        # Get probabilities
        probabilities = None
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(df)[0]
            print(f"\n📊 PROBABILITIES:")
            category_names = ["Clothing", "Accessories", "Footwear", "Outerwear"]
            for i, (name, p) in enumerate(zip(category_names, proba)):
                print(f"  {i} - {name}: {p*100:.2f}%")
            
            category_colors = {0: "#198754", 1: "#0d6efd", 2: "#ffc107", 3: "#6f42c1"}
            probabilities = [
                {
                    "category": category_names[i],
                    "probability": float(proba[i] * 100),
                    "color": category_colors[i]
                }
                for i in range(len(proba))
            ]
        
        print(f"\n🎯 PREDICTED: Category {predicted_category} ({category_names[predicted_category]})")
        print("="*60 + "\n")
        
        category_labels = {
            0: "Clothing",
            1: "Accessories", 
            2: "Footwear",
            3: "Outerwear"
        }
        
        return {
            "predicted_category": predicted_category,
            "category_label": category_labels.get(predicted_category, f"Category {predicted_category}"),
            "probabilities": probabilities,
            "status": "success"
        }
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/")
def root():
    return {"message": "Customer Behavior Prediction API", "status": "online"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": clf is not None}