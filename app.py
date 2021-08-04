import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pickle
import streamlit as st

pickle_in = open("model.pkl","rb")
regressor = pickle.load(pickle_in)

def index():
    return "Congratulations, it's a web app!"

def lemmat(text):
    lemma=WordNetLemmatizer()
    words=word_tokenize(text)
    return ' '.join([lemma.lemmatize(word) for word in words])

def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def getpolarity(text):
    return TextBlob(text).sentiment.polarity

def predict_price(Open,High,Low,Volume,Headlines):
    df2 = pd.DataFrame()
    df2['Volume']=Volume,Volume
    df2['Open']=Open,Open
    df2['High']=High,High
    df2['Low']=Low,Low
    df2['Headlines'] = Headlines
    df2['Headlines'] = df2['Headlines'].astype(str)
    df2.Headlines.replace("[^a-zA-Z]"," ",regex=True,inplace=True)
    df2['Headlines'] = df2['Headlines'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    stop = stopwords.words('english')
    df2['Headlines'] = df2['Headlines'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    stq = PorterStemmer()
    df2['Headlines'] = df2['Headlines'].apply(lambda x: " ".join([stq.stem(word) for word in x.split()]))
    df2['Headlines'] = df2['Headlines'].apply(lemmat)
    analyzer = SentimentIntensityAnalyzer()
    df2['compound'] = df2['Headlines'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df2['negative'] = df2['Headlines'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
    df2['neutral'] = df2['Headlines'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
    df2['positive'] = df2['Headlines'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
    df2['Subjectivity']=df2['Headlines'].apply(getsubjectivity)
    df2['Polarity']=df2['Headlines'].apply(getpolarity)

    df2_testdata= df2[['Volume','Open','High','Low','compound', 'negative', 'neutral', 'positive','Subjectivity','Polarity']]
    return df2_testdata

def main():
    st.title("MICROSOFT CORPORATION STOCK PRICE PREDICTION")

    html_temp = """
    <div style="background-color:rgb(0, 238, 255);padding:10px">
    <div style="background-color:#FFD700;padding:10px">
    <h2 style="color:white;text-align:center;">Microsoft Stock Daily Close Price Predictor</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Open = st.text_input("Open","Enter a value")
    High = st.text_input("High","Enter a value")
    Low = st.text_input("Low","Enter a value")
    Volume = st.text_input("Volume","Enter a value")
    Headlines= st.text_input("Headlines","Enter a news headline")
    df3=pd.DataFrame()
    df3=predict_price(Open,High,Low,Volume,Headlines)
    df4=np.array(df3)
    result=''
    if st.button("Predict"):
        result = regressor.predict(df4)[0]
    st.success('Predicted Close Price : $ {}'.format(result))
    if st.button("About"):
        st.text("Close price of a stock is predicted based on Open, High, Low prices of the stock, Volume of \n the Stock and news headlines using Linear Regression")
        st.text("Inputs \nOpen: Open Price of a stock")
        st.text("High: Highest Price of a stock")
        st.text("Low: Lowest Price of a stock")
        st.text("Volume: Volume Available for a stock")
        st.text("Headlines: Recent news headlines available regarding the stock")
if __name__ =='__main__':
    main()
