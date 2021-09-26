import streamlit as st
import numpy as np
import joblib
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: black;
    color : white;
}
</style>""", unsafe_allow_html=True)
import pandas as pd

df = pd.read_csv("data/final_dataset.csv")
location_df = df.drop(['total_sqft','bath','bhk','price'],axis = "columns")
X = df.drop("price",axis = "columns")
#st.write(type(locations.columns.values)) # we got numpy array
locations = location_df.columns.values
locations = np.append(locations,["Other"])
model = joblib.load("data/price_model.joblib")

def predict_price(location,total_sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]


model_score = pd.read_csv('data/model_scores.csv')


# st.write(df.head(10)) 
#st.write(locations.columns)


# streamlit run main.py
# containers vs columns
# containers create sections in vertical way one after another
# columns create sections side by side like in bootstrap

header = st.container()
model_prediction = st.container()

dataset = st.container()

features_used = st.container()
features_created = st.container()
model_selection = st.container()
final_model_results = st.container()

with model_prediction:
    # st.header("Predict Your Ideal Home Price")
    form = st.form(key = "prediction_form")
    sqft_area = form.number_input("Area(Square Feet)")
    location = form.selectbox("Location",locations)
    bath = form.selectbox("Bath",[1,2,3,4,5])
    bed_rooms = form.selectbox("Bed Rooms",[1,2,3,4,5])
    submit = form.form_submit_button("Predict Price")

    if submit:
        price = predict_price(location,sqft_area, bath,bed_rooms)
        st.text(f"Estimated Price:  {round(price,2)} Lakh")

with header:
    st.title("Bengaluru House Price Estimation")
    #st.text("In this project I will predict house prices in Bengaluru")

with dataset:
    st.header("Bengaluru House price dataset")
    st.text("I found this dataset on : https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data")
    dataset = pd.read_csv("data/bengaluru_house_prices.csv")
    st.write(dataset.head())

with features_used:
    st.header("Features Used")
    st.markdown("* **Total square feet:** Total Area of house in square feet")
    st.markdown("* **Bath:** Number of bathrooms in the house")
    st.markdown("* **Size:** Number of bedrooms,hall,kitchen in the house ")
    st.markdown("* **Location:** Place where house is located ")


with features_created:
    st.header("Features Engineering")
    st.markdown("* **Bhk:** Created bhk feature from size by splitting to get only the number of bhk without string ")

    st.markdown("* **Price per sqft:** It is created by using this logic (df.price * 100000) / df.total_sqft to get price per sqft")

with model_selection:
    st.text("After Data cleaning and outlier removal using z score we will now select the best model and its hyper parameters using Gridsearch cv")
    st.header("Selecting Best Model")
    st.text("I was planning to use gridsearchcv to find best model and best hyper parameters but Since my computer was taking too long to run gridsearch so I tested every model on train and test dataset  ")
    st.subheader("Here are the results")
    st.write(model_score)

with final_model_results:
    st.header("R2 score using Linear Regression model")
    st.text("r2 score : 0.79" )


    

