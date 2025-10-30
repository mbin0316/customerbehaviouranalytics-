#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

behaviour_df=pd.read_csv("cleaned_processed_customer_data.csv")

behaviour_df.head()


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
import numpy as np

# --- Step 1: Copy dataset ---
df = behaviour_df.copy()


# --- Step 2: Target encoding ---
df['category_encoded'] = df['category'].astype('category').cat.codes

# --- Step 3: ENHANCED Feature engineering for buying behavior prediction ---
# Purchase behavior features
df['avg_spent_per_day'] = df['purchase_amount'] / (df['purchase_frequency_days'] + 1)
df['is_high_spender'] = (df['purchase_amount'] > df['purchase_amount'].quantile(0.75)).astype(int)
df['price_per_frequency'] = df['purchase_amount'] / (df['purchase_frequency_days'] + 1)
df['is_frequent_buyer'] = (df['purchase_frequency_days'] < df['purchase_frequency_days'].quantile(0.25)).astype(int)

# Value-based segmentation
df['customer_value'] = df['purchase_amount'] * (30 / (df['purchase_frequency_days'] + 1))
df['high_value_customer'] = (df['customer_value'] > df['customer_value'].quantile(0.75)).astype(int)

# Discount sensitivity
df['discount_user'] = df['discount_applied'].map({'Yes': 1, 'No': 0})
df['promo_user'] = df['promo_code_used'].map({'Yes': 1, 'No': 0})
df['deal_seeker'] = ((df['discount_user'] == 1) | (df['promo_user'] == 1)).astype(int)

# Loyalty indicators
df['is_subscriber'] = df['subscription_status'].map({'Yes': 1, 'No': 0})
df['loyalty_score'] = df['is_subscriber'] + df['is_frequent_buyer'] + (df['review_rating'] >= 4).astype(int)

# Strategic interaction features for buying prediction
df['gender_season'] = df['gender'] + '_' + df['season']
df['age_group_frequency'] = df['age_group'] + '_' + df['frequency_of_purchases']

# --- Step 4: Define X and y ---
X = df.drop(columns=['category', 'category_encoded', 'customer_id', 'item_purchased'])
y = df['category_encoded']

# Check class balance BEFORE SMOTE
print("="*60)
print("CLASS DISTRIBUTION (BEFORE SMOTE)")
print("="*60)
print(y.value_counts().sort_index())
print(f"\nTotal samples: {len(y)}")
print("\n")

categorical_cols = [
    'gender',           
    'age_group',        
    'size',             
    'color',            
    'season',           
    'frequency_of_purchases',  
    'gender_season',    
    'age_group_frequency',  
]

# High-value numeric features for purchase prediction
numeric_cols = [
    'purchase_amount',        
    'review_rating',          
    'purchase_frequency_days',
    'age',                    
    'customer_value',         
    'loyalty_score',         
    'is_high_spender',        
    'is_frequent_buyer',      
    'high_value_customer',    
    'deal_seeker',            
    'is_subscriber',          
    'discount_user',          
    'promo_user',             
]

print("Selected Features:")
print(f"Categorical: {len(categorical_cols)} features")
print(f"Numeric: {len(numeric_cols)} features")
print(f"Total: {len(categorical_cols) + len(numeric_cols)} features\n")

# --- Step 5: Preprocessor ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# --- Step 6: Stratified train-test split (BEFORE SMOTE) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*60)
print("TRAIN SET CLASS DISTRIBUTION (BEFORE SMOTE)")
print("="*60)
print(pd.Series(y_train).value_counts().sort_index())
print(f"\nTrain samples: {len(y_train)}")
print("\n")

# --- Step 7: XGBoost Model 
xgb_model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    reg_alpha=0.1,
    reg_lambda=1.5,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist'
)

# --- Step 8: Build pipeline with SMOTE ---
clf = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),  # SMOTE after preprocessing
    ('feature_selection', SelectKBest(mutual_info_classif, k='all')),
    ('model', xgb_model)
])

# --- Step 9: Train model ---
print("="*60)
print("APPLYING SMOTE AND TRAINING MODEL...")
print("="*60)
clf.fit(X_train, y_train)
print("Training complete!\n")

# Show class distribution after SMOTE (indirectly, by checking what SMOTE would produce)
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

print("="*60)
print("TRAIN SET CLASS DISTRIBUTION (AFTER SMOTE)")
print("="*60)
print(pd.Series(y_train_resampled).value_counts().sort_index())
print(f"\nResampled train samples: {len(y_train_resampled)}")
print("\n")

# --- Step 10: Evaluate ---
y_pred = clf.predict(X_test)

print("="*60)
print("MODEL EVALUATION (ON ORIGINAL TEST SET)")
print("="*60)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Weighted F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Macro F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=df['category'].unique()))

# Cross-validation score
print("="*60)
print("CROSS-VALIDATION (WITH SMOTE)")
print("="*60)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"Cross-validation F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance analysis
print("\n" + "="*60)
print("TOP 20 IMPORTANT FEATURES FOR PREDICTING NEXT PURCHASE")
print("="*60)
feature_names = clf.named_steps['preprocessor'].get_feature_names_out()
importances = clf.named_steps['model'].feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df.head(20))



# In[ ]:


behaviour_df.isnull().sum()


# In[ ]:


import joblib

# Save  trained pipeline as
joblib.dump(clf, "customer_behavior_model.pkl")





