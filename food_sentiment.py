import streamlit as st
import joblib
import pandas as pd
import numpy as np

vectorizer=joblib.load("vectorizer.pkl")
model=joblib.load("Sentiment_model.pkl")

st.set_page_config(layout="wide")
st.sidebar.title("About project")
st.sidebar.write("- objective of this project is to predict sentiment(Neg/Pos)of food review")
st.sidebar.title("Libraries")
st.sidebar.markdown("""
- Pandas
- Numpy
- sklearn
""")
st.sidebar.title("Cloud")
st.sidebar.markdown("Streamlit")
st.sidebar.title("Contact Us")
st.sidebar.markdown("9999876545")
# Add CSS for banner
st.markdown("""
<style>
.banner {
        background-color: #ff914d;
        background-position: center;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 40px;
}
</style>
<div class="banner">🍔 Food Sentiment Analysis</div>
""",unsafe_allow_html=True)

st.write("\n" )

col1,col2 = st.columns([.4,.6])
with col1:
    st.header("Predict single review")
    review = st.text_input("Enter Review") 
    if st.button("Predict"):
        X_test = vectorizer.transform([review])
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)
        if pred[0] == 0:
            st.error("**Sentiment = Negative**")
            st.warning(f"Confidence Score = {prob[0][0]:.2f}")
        else:
            st.success("**Sentiment = Positive**")
            st.warning(f"Confidence Score = {prob[0][1]:.2f}") 
with col2:
    st.header("Predict Bulk review")
    file = st.file_uploader("Select a file",type =["csv","txt"])
    if file:
        df = pd.read_csv(file,header=None,names=["Review"])
        placeholder = st.empty()
        placeholder.dataframe(df)
        if st.button("Bulk Prediction"):
            X_test = vectorizer.transform(df.Review)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
            Sentiment = ["Positive" if i==1 else "Negative" for i in pred]
            df['sentiment'] = Sentiment
            df['Confidence'] = np.max(prob,axis = 1)
            placeholder.dataframe(df)
        
            