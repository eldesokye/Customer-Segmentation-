from flask import Flask, render_template, request , jsonify
import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json 



app = Flask(__name__)
model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


def load_and_clean_data(file_path):
    # Load the data
    retail = pd.read_csv(file_path , sep=',' , encoding='ISO-8859-1' , header=0)

    # convert Customer ID to String and crate Amount column 
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity'] * retail['PriceEach']

    # compute RFM metrics 
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f= retail.groupby('CustomerID')['InvoiceDate'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
    max_date = retail['InvoiceDate'].max()
    retail['Diff']= (max_date - retail['InvoiceDate']).dt.days
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p["Diff"]= rfm_p["Diff"].dt.days
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID' , how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID' , how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    # Removing the (statistical)  outliers for amount
    Q1 = rfm.Monetary.quantile(0.05)
    Q3 =rfm.Monetary.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Monetary >= Q1 - 1.5 * IQR) & (rfm.Monetary <= Q3 + 1.5 * IQR)]


    # Removing the (statistical) outliers for Recency
    Q1 = rfm.Recency.quantile(0.05)
    Q3 =rfm.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Recency >= Q1 - 1.5 * IQR) & (rfm.Recency <= Q3 + 1.5 * IQR)]

    # Removing the (statistical) outliers for Frequency 
    Q1 = rfm.Frequency.quantile(0.05)
    Q3 =rfm.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5 * IQR) & (rfm.Frequency <= Q3 + 1.5 * IQR)]

    return rfm 


def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    # Define the features and target variable
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    # instantiate 
    scaler = StandardScaler()
    # fit the scaler on the data
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

    return rfm_df_scaled



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd, file.filename)
    file.save(file_path)
    df = preprocess_data(file_path)[1]
    results_df = model.predict(df)
    df_with_id = preprocess_data(file_path)[0]

    df_with_id['Cluster_Id'] = results_df

    # Generate the images and save them 

    sns.stripplot(x='Cluster_Id', y='Amount', data=df_with_id , hue= df_with_id['Cluster_Id'])
    amount_img_path = 'static/Amountpplot.png'
    plt.savefig(amount_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Frequency', data=df_with_id , hue= df_with_id['Cluster_Id'])
    frequency_img_path = 'static/Cluster_Idfrequency.png'
    plt.savefig(frequency_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Recency', data=df_with_id , hue= df_with_id['Cluster_Id'])
    recency_img_path = 'static/Cluster_IdRecency.png'
    plt.savefig(recency_img_path)
    plt.clf()


    # return the filenames of the generated images as a json response 
    response = {
        'amount_img': amount_img_path,
        'frequency_img': frequency_img_path,
        'recency_img': recency_img_path
    }
    return json.dumps(response)



if __name__ == '__main__':
    app.run(debug=True)