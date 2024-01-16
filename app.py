import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.model_selection import train_test_split


clf = None
X, y = make_moons(500, noise=0.30, random_state=402)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=402
)

st.sidebar.markdown("# Bagging Classifier")
# st.sidebar.text("Select Base Estimator")
model = st.sidebar.selectbox(
    "Select Base Estimator",
    ("Support Vector Classifier", "Decision Tree", "Naive Bayes"),
)

n_estimators = st.sidebar.number_input("Number of Estimators", 100, 400)

max_samples = st.sidebar.slider("Max Samples", 100, 300)

bootstrap_samples = (
    True
    if st.sidebar.radio("Bootstrap Samples", options=("True", "False")) == "True"
    else False
)

n_features = st.sidebar.slider("No of Features", 1, 2)

bootstrap_features = (
    True
    if st.sidebar.radio("Bootstrap Features", options=("True", "False")) == "True"
    else False
)

# Load initial Graph
fig, ax = plt.subplots()

# Plot initial Graph
ax.scatter(X.T[0], X.T[1], c=y, cmap="rainbow")
orig = st.pyplot(fig)

if st.sidebar.button("Lets Go!"):
    # estimator will be used to train BaggingClassifier
    # clf (classifier) will be used to train local model
    if model == "Naive Bayes":
        estimator = GaussianNB()
        clf = GaussianNB()
    elif model == "Decision Tree":
        estimator = DecisionTreeClassifier()
        clf = DecisionTreeClassifier()
    elif model == "Support Vector Classifier":
        estimator = SVC()
        clf = SVC()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    bag_clf = BaggingClassifier(
        estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=bootstrap_samples,
        bootstrap_features=bootstrap_features,
        random_state=404,
    )
    bag_clf.fit(X_train, y_train)
    y_pred_bagg = bag_clf.predict(X_test)

    orig.empty()

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    col1, col2 = st.columns(2)
    with col1:
        st.header(estimator)
        ax.scatter(X.T[0], X.T[1], c=y, cmap="rainbow")
        orig = st.pyplot(fig)
        st.subheader(
            "Accuracy for "
            + str(estimator)
            + "="
            + str(round(accuracy_score(y_test, y_pred), 2))
        )

    with col2:
        st.header("Bagging Classifier")
        ax1.scatter(X.T[0], X.T[1], c=y, cmap="rainbow")
        orig1 = st.pyplot(fig1)
        st.subheader(
            "Accuracy for Bagging = "
            + str(round(accuracy_score(y_test, y_pred_bagg), 2))
        )
