import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Streamlit Machine Learning App")

numPrevOwners = st.number_input("Enter numPrevOwners")
age = st.number_input("Enter Age")
basement = st.number_input("Enter Basement")
numberOfRooms = st.number_input("Enter numberOfRooms")
isNewBuilt = st.number_input("Enter isNewBuilt")
squareMeters = st.number_input("Enter squareMeters")
floors = st.number_input("Enter floors")
cityPartRange = st.number_input("Enter cityPartRange")
countFac = st.number_input("Enter countFac")
garage = st.number_input("Enter garage")
hasGuestRoom = st.number_input("Enter hasGuestRoom")
attic = st.number_input("Enter attic")

preds = [
    'numPrevOwners', 'age', 'basement', 'numberOfRooms', 'isNewBuilt',
    'squareMeters', 'floors', 'cityPartRange', 'countFac', 'garage',
    'hasGuestRoom', 'attic'
]

# If button is pressed
if st.button("Submit"):

    # Unpickle classifier
    clf = joblib.load("clf.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[
        numPrevOwners, age, basement, numberOfRooms, isNewBuilt, squareMeters,
        floors, cityPartRange, countFac, garage, hasGuestRoom, attic
    ]],
                     columns=preds)

    # Get prediction
    prediction = clf.predict(X)[0]

    # Output prediction
    st.text(f"This instance is a {prediction}")
