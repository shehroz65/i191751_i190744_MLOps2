#!/usr/bin/env python
# coding: utf-8

# In[20]:


from flask import Flask, render_template, request
import pickle
from sklearn import preprocessing
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import math
import pandas as pd
app = Flask(__name__)


# In[25]:


def RMSE(l1, l2):
    l1 = [1,2,3,4,5]
    l2 = [1.6,2.5,2.9,3,4.1]

    MSE = np.square(np.subtract(l1,l2)).mean() 

    RMSE = math.sqrt(MSE)
    
    return RMSE


# In[21]:


def get_live_data(sd, ed):
    df = yf.download("AAPL", start = sd, end=ed)
    df = df.reset_index().rename(columns={'date': 'new_column'})
    df.drop(columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1, inplace = True)
    df.dropna(inplace = True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract year, month, day, and weekday features from the Date column
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday

    # Weekday is represented as an integer, so you can convert it to a string if needed
    df['Weekday'] = df['Weekday'].apply(lambda x: pd.Timestamp(2022, 1, x+1).strftime("%A"))
    # Import label encoder
  
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()

    # Encode labels in column 'species'.
    df['Weekday']= label_encoder.fit_transform(df['Weekday'])

    df.drop(columns = ['Date'], axis = 1, inplace = True)
    
    return df


# In[18]:



# Load the pickled model from file
with open('model.pkl', 'rb') as f:
model = pickle.load(f)


# In[26]:


@app.route('/')
def dashboard():
    start_date = '2016-01-01'
    end_date = datetime.today().strftime("%Y-%m-%d")
    # get live data
    data = get_live_data(start_date, end_date)
    #print(data)
    data = data.drop(columns = ['Close'], axis = 1)
    # make predictions using the model
    pred = model.predict(data)
    # calculate accuracy
    RMSE = RMSE(list(data['Close']), pred)
    # render the dashboard template with the results
    # , data=data, pred=pred, acc=acc
    return render_template('dashboard.html', data=data, pred=pred, acc=RMSE)


# In[ ]:


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)

   # app.run(debug=True)

