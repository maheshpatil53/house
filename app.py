import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request,jsonify
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import sqlite3
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
prompt = open('website_text.txt', 'r',encoding='utf-8').read()
property_assistant_template = prompt + """
You are "Mr. Mahesh". 
Your expertise is exclusively in providing information and advice about anything related to Real Estate in India. 
This includes any general property related queries. 
You do not provide information outside of this scope.
Question: {question} 
Answer: 
"""

property_assistant_template = PromptTemplate( 
    input_variables=["question"], 
    template=property_assistant_template 
    ) 

llm = OpenAI(api_key=os.getenv("api_key")) 

llm_chain = property_assistant_template | llm 

def query_llm(question): 
    response=(llm_chain.invoke({'question': question}))
    return response

app = Flask(__name__)
df=pd.read_csv("dataframe/df")
df = df[['total_sqft', 'size', 'site_location','price']]

@app.route("/chatbot", methods=["POST"]) 
def chatbot(): 
    data = request.get_json() 
    question = data["question"] 
    response = query_llm(question) 
    return jsonify({"response": response})

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method=='POST':
        v1=int(request.form['area'])
        v2=int(request.form['bedroom'])
        v3=request.form['Area']
        conn = sqlite3.connect('Sample.db')
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO HPP (area_sqft, bedrooms, city_area) VALUES (?, ?, ?)""", (v1,v2,v3))
        conn.commit()
        new_input={'total_sqft':v1,'size':v2,'site_location':v3}
        new_input_df = pd.DataFrame([new_input])
        new_input_df = new_input_df.convert_dtypes()
        numeric_col= df.select_dtypes(include=['int']).columns.tolist()
        categorical_cols = df.select_dtypes('object').columns.tolist()
        enc = preprocessing.OneHotEncoder()
        enc.fit(df[['site_location']])
        one_hot = enc.transform(df[['site_location']]).toarray()
        encoded_cols = list(enc.get_feature_names_out(categorical_cols))
        df[encoded_cols]=one_hot
        X=df[numeric_col+encoded_cols].to_numpy()
        y=df['price'].to_numpy()
        model = LinearRegression()
        model.fit(X, y)
        new_input_df[encoded_cols] = enc.transform(new_input_df[['site_location']]).toarray()
        new_input_df=new_input_df[numeric_col+encoded_cols].to_numpy()
        prediction = model.predict(new_input_df)[0]
        rounded_prediction = round(prediction, 2)
        return render_template('index.html', prediction=rounded_prediction)
    else:
        return render_template('index.html')
  
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)