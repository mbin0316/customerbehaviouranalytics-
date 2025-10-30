#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd




behaviour_df=pd.read_csv("dataset/customer_shopping_behavior.csv")




# In[38]:


behaviour_df.info()


# In[39]:


behaviour_df.describe()


# In[40]:


behaviour_df.isnull().sum()


# In[41]:


behaviour_df['Review Rating']=behaviour_df.groupby('Category')['Review Rating'].transform(lambda x:x.fillna(x.median()))


# In[42]:


behaviour_df.columns=behaviour_df.columns.str.lower()
behaviour_df.columns=behaviour_df.columns.str.replace(' ','_')


# In[43]:


behaviour_df.columns


# In[44]:


behaviour_df= behaviour_df.rename(columns={'purchase_amount_(usd)': 'purchase_amount'})


# In[45]:


behaviour_df.columns


# In[46]:


#Feature engineering

#1)Create new column for age group
labels=["Young Adult","Adult","Middle-Aged","Elderly"]
behaviour_df['age_group']=pd.qcut(behaviour_df['age'],q=4,labels=labels)


behaviour_df.head(20)



# In[47]:


#2)Create a column of purchase frequency days

days = {
    'Fortnightly': 14,
    'Weekly': 7,
    'Monthly': 30,
    'Quarterly': 90,
    'Bi-Weekly': 14,
    'Annually': 365,
    'Every 3 Months': 90
}
behaviour_df['purchase_frequency_days']= behaviour_df['frequency_of_purchases'].map(days)


# In[48]:


behaviour_df.head(10)


# In[49]:


get_ipython().system('pip install psycopg2-binary sqlalchemy')


# In[50]:


import pandas as pd
from sqlalchemy import create_engine


engine = create_engine("postgresql+psycopg2://postgres:password@localhost/postgres")



# --- Step 2: Write DataFrame to PostgreSQL ---
# This will create a table named 'people' in the 'public' schema
behaviour_df.to_sql("customer", engine, if_exists="replace", index=False)
print("\n✅ DataFrame successfully written to PostgreSQL table 'cusstomer'.")

#Step 3: Read back from PostgreSQL ---
query = "SELECT * FROM customer;"
df_from_db = pd.read_sql(query, engine)
print("\n✅ Data fetched back from PostgreSQL:")
print(df_from_db)


# In[ ]:





# In[51]:


# Save preprocessed DataFrame as CSV
behaviour_df.to_csv("cleaned_processed_customer_data.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




