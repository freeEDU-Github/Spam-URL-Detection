import sys
import pandas as pd
import numpy as np
import glob
import sklearn
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
import pickle
import joblib
import streamlit as st


def extractUrl(data):
    url = str(data)
    extractSlash = url.split('/')
    result = []

    for i in extractSlash:
        extractDash = str(i).split('-')
        dotExtract = []

        for j in range(0, len(extractDash)):
            extractDot = str(extractDash[j]).split('.')
            dotExtract += extractDot

        result += extractDash + dotExtract
    result = list(set(result))

    return result


def main():
    st.title("Spam URL Analysis with a Machine Learning Approach")

    st.subheader("Spam URL Analysis")
    st.markdown("Backlinks that are placed on pages and websites regardless of context or user experience in an attempt to boost search rankings are referred to as spam URLs. Search engines have strict policies against link spam and will penalize or devalue spammy links. ")

    image = Image.open('spam url.jpg')
    st.image(image, caption='Spam URL')

    st.subheader(
        "The primary goal of this project is to identify whether the URLs are spam or not")

    url = pd.read_csv("url_spam_classification.csv")
    sample = pd.read_csv("sample data.csv")
    st.dataframe(sample)

    user_input = st.text_input("Give me some URL to work on : ")
    user_input = np.array([user_input])
    url['is_spam'] = url.is_spam.apply(str)
    url['is_spam'] = url['is_spam'].apply(lambda x: 1 if x == "True" in x else 0)

    X = url["url"]
    y = url["is_spam"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    cv = CountVectorizer(tokenizer=extractUrl)
    trainCV = cv.fit_transform(X_train)
    testCV = cv.transform(X_test)
    userCV = cv.transform(user_input)
    dtModel = tree.DecisionTreeClassifier()
    dtModel.fit(trainCV, y_train)
    prediction = dtModel.predict(userCV)

    if prediction == 0:
        st.success("This is not a spam")

    else:
        st.error("🚨️ALERT -  THIS IS A SPAM! 🚨")



if __name__ == '__main__':
    main()