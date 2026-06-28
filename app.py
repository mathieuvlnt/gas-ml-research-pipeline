import streamlit as st
import plotly.graph_objects as go

from market_data import MarketData
from features import FeatureEngine
from dataset import DatasetBuilder
from model import ModelTrainer
from evaluation import ModelEvaluator

st.set_page_config(page_title="European Gas Market ML Research Pipeline", layout="wide")

st.title("European Gas Market ML Research Pipeline")

# Sidebar
st.sidebar.header("Research Settings")

experiment = st.sidebar.selectbox("Research Experiment",["Direction Prediction",
                                                         "Large Move Prediction",
                                                         "High Volatility Prediction"])

model_choice = st.sidebar.selectbox("Model",["Random Forest", "Logistic Regression"])

threshold = 0.02

if experiment == "Large Move Prediction":
    threshold = st.sidebar.slider(
        "Large Move Threshold",
        min_value=0.01,
        max_value=0.05,
        value=0.02,
        step=0.01)

history = st.sidebar.selectbox("Historical Window",["Last 3 Years",
                                                    "Last 5 Years",
                                                    "Last 10 Years"])

if history == "Last 3 Years":
    start_date = "2023-01-01"

elif history == "Last 5 Years":
    start_date = "2021-01-01"

elif history == "Last 10 Years":
    start_date = "2016-01-01"


# Pipeline
def run_pipeline(experiment, threshold, start_date, model_choice):

    tickers = {"TTF": "TTF=F",
               "BRENT": "BZ=F"}

    market = MarketData(tickers, start_date)
    prices = market.load_prices()

    feature_engine = FeatureEngine(prices)
    feature_data = feature_engine.compute_features()

    if experiment == "Direction Prediction":
        target_type = "direction"
    elif experiment == "Large Move Prediction":
        target_type = "large_move"
    elif experiment == "High Volatility Prediction":
        target_type = "high_volatility"

    dataset = DatasetBuilder(
        feature_data,
        prices,
        target_type=target_type,
        threshold=threshold)

    X_train, X_test, y_train, y_test = dataset.build_dataset()

    trainer = ModelTrainer(model_choice)
    trainer.fit(X_train, y_train)

    predictions = trainer.predict(X_test)
    probabilities = trainer.predict_proba(X_test)

    evaluator = ModelEvaluator(y_test, predictions)
    metrics = evaluator.get_metrics()

    importance = trainer.get_feature_importance(X_train.columns)

    return prices, feature_data, probabilities, importance, metrics, evaluator


# Run app
with st.spinner("Running machine learning experimente..."):
    prices, feature_data, probabilities, importance, metrics, evaluator = run_pipeline(
        experiment,
        threshold,
        start_date,
        model_choice)

    st.markdown("""
    **Mathieu Voluntario** 
    
    Programme Grande École (PGE)  
    Data Science & Finance Track  
    EDHEC Business School
    """)
    
    colA, colB = st.columns(2)
    with colA:
        st.info(f" Experiment : {experiment}")
    
    with colB:
        st.info(f" Model : {model_choice}")
        
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    col2.metric("Precision", f"{metrics['Precision']:.1%}")
    col3.metric("Recall", f"{metrics['Recall']:.1%}")

    show_matrix = st.checkbox("Show Confusion Matrix")
    if show_matrix:
        st.dataframe(evaluator.get_confusion_matrix())
    
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Market Data", "Features", "Predictions","Feature Importance"])

    # Prices
    with tab1:
        st.subheader("Market Prices")
        price_fig = go.Figure()
        for asset in prices.columns:
            price_fig.add_trace(go.Scatter(
                x=prices.index,
                y=prices[asset],
                name=asset))
        
        price_fig.update_layout(
            template="plotly_white",
            height=450,
            xaxis_title="Date",
            yaxis_title="Price")
        
        st.plotly_chart(price_fig, use_container_width=True)

    # Volatility and Z-score
    with tab2: 
        st.subheader("TTF Volatility and Z-Score")
        feature_fig = go.Figure()
        feature_fig.add_trace(go.Scatter(
            x=feature_data.index,
            y=feature_data["Volatility"],
            name="Volatility")
                             )
        
        feature_fig.add_trace(go.Scatter(
            x=feature_data.index,
            y=feature_data["TTF Z-Score"],
            name="TTF Z-Score")
                             )
        
        feature_fig.update_layout(
            template="plotly_white",
            height=450,
            xaxis_title="Date")
        
        st.plotly_chart(feature_fig, use_container_width=True)

    # Probabilities
    with tab3:
        st.subheader("Predicted Probability")
        probability_fig = go.Figure()
        probability_fig.add_trace(go.Scatter(
            x=probabilities.index,
            y=probabilities,
            name="Predicted Probability"))
        
        probability_fig.add_hline(
            y=0.5,
            line_dash="dash")
        
        probability_fig.update_layout(
            template="plotly_white",
            height=450,
            xaxis_title="Date",
            yaxis_title="Probability")
        
        st.plotly_chart(probability_fig, use_container_width=True)

    # Feature importance
    with tab4:
        st.subheader("Feature Importance")
        importance_fig = go.Figure()
        importance_fig.add_trace(go.Bar(
            x=importance.index,
            y=importance.values,
            name="Importance"))
        
        importance_fig.update_layout(
            template="plotly_white",
            height=450,
            xaxis_title="Feature",
            yaxis_title="Importance",
            xaxis_tickangle=-45)
        
        st.plotly_chart(importance_fig, use_container_width=True)