from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import json


app = Flask(__name__)
model = pickle.load(open('kmeans_model.pkl','rb'))


def load_and_clean_data(file_path):

    print("data load function running")

    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity']*retail['UnitPrice']

    # Compute RFM metrices
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d/%m/%Y %H:%M')
    max_date = max(retail['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    # Remove outliers
    rfm_numeric = rfm.select_dtypes(include=['int64', 'float64'])
    Q1 = rfm_numeric.quantile(0.05)
    Q3 = rfm_numeric.quantile(0.95)
    IQR = Q3 - Q1
    
    rfm = rfm[(rfm_numeric.Amount >= Q1[0] - 1.5*IQR[0]) & (rfm.Amount <= Q3[0] + 1.5*IQR[0])]
    rfm = rfm[(rfm_numeric.Recency >= Q1[2] - 1.5*IQR[2]) & (rfm.Recency <= Q3[2] + 1.5*IQR[2])]
    rfm = rfm[(rfm_numeric.Frequency >= Q1[1] - 1.5*IQR[1]) & (rfm.Frequency <= Q3[1] + 1.5*IQR[1])]

    return rfm

def preprocess_data(file_path):
    print("preproccess function running")
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    # Instantiate
    scaler = StandardScaler()
    # fit_transform
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    # rfm_df_scaled
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
    return rfm,rfm_df_scaled




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/most_selling_product', methods=['POST'])
def most_selling_product():
    print("Most selling product function running")
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)

    # Load the dataset
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)
    
    # Group by product description and sum quantities
    product_sales = retail.groupby('Description')['Quantity'].sum().reset_index()
    
    # Find the most-selling product
    most_selling = product_sales.sort_values(by='Quantity', ascending=False).iloc[0]
    product_name = most_selling['Description']
    quantity_sold = int(most_selling['Quantity'])  # Convert to Python int

    #Filter rows with negative values
    product_sales=product_sales[product_sales['Quantity']>0]

    # Find the least-selling product
    least_selling = product_sales.sort_values(by='Quantity', ascending=True).iloc[0]
    lproduct_name = least_selling['Description']
    lquantity_sold = int(least_selling['Quantity'])  # Convert to Python int


    print(f"Most selling product: {product_name} ({quantity_sold} units)")
    print(f"Least selling product: {lproduct_name} ({lquantity_sold} units)")
    # Return the result as JSON
    response = {
        "product_name": product_name,
        "quantity_sold": quantity_sold,  # Native Python type
        "lproduct_name": lproduct_name,
        "lquantity_sold": lquantity_sold  # Native Python type        
    }
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    print("predict function running")
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)
    df = preprocess_data(file_path)[1]
    results_df = model.predict(df)
    #results_df = pd.DataFrame(results_df)
    df_with_id = preprocess_data(file_path)[0]
    
    df_with_id['Cluster_Id'] = results_df

    # Save the clustered data to a CSV file
    csv_path = 'static/final_clustered_customers.csv'
    df_with_id.to_csv(csv_path, index=False)

    # Calculate number of customers per Cluster_Id
    cluster_counts = df_with_id['Cluster_Id'].value_counts().to_dict()
    count_0 = cluster_counts.get(0, 0)
    count_1 = cluster_counts.get(1, 0)
    count_2 = cluster_counts.get(2, 0)

    print("##-----Values Printed-----##")
    print(df_with_id)
    
    #Generate HTML Table
    df_html= df_with_id.to_html(classes='table table-bordered table-striped', index=False)

    #Generate images and save them
    sns.stripplot(x='Cluster_Id', y='Amount', data=df_with_id, hue='Cluster_Id')
    amount_img_path = 'static/ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Frequency', data=df_with_id, hue='Cluster_Id')
    freq_img_path = 'static/ClusterId_Frequency.png'
    plt.savefig(freq_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Recency', data=df_with_id, hue='Cluster_Id')
    recency_img_path = 'static/ClusterId_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()

    #return the filenames of the generated images as a JSON response
    response = {'amount_img' : amount_img_path,
                'freq_img' : freq_img_path,
                'recency_img' : recency_img_path,
                'table_html' : df_html,
                'csv_file': csv_path,
                'count_0': count_0,
                'count_1': count_1,
                'count_2': count_2 }
    return json.dumps(response)



# for local
if __name__=="__main__":
    app.run(debug=True)

# for cloud
# if __name__ == "__main__":
#   app.run(host = '0.0.0.0', port=8000)    