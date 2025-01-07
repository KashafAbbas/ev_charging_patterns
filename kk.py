# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="EV Charging Dashboard", page_icon="⚡", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('ev_charging_patterns.csv')  
    return data

data = load_data()

# Title
st.title("⚡ Electric Vehicle Charging Patterns: Comprehensive EDA")

# Sidebar
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Choose a section:", ["🏠 Introduction", "📊 Summary & Insights", "📈 Visualizations", "🔮 Prediction", "📚 Conclusion"])
st.sidebar.write("\n---")
st.sidebar.write("🚀 Developed by Kashaf Abbas")
# Introduction
if page == "🏠 Introduction":
    st.header("🏠 Introduction")
    st.write("Welcome to the Electric Vehicle Charging Patterns Dashboard! This app provides a detailed exploration of EV charging behaviors through data analysis, interactive visualizations, and predictive modeling.")
    st.write("### 🚀 Features of the App:")
    st.write("- 📊 **Summary & Insights**: Explore key statistics and correlations in the dataset.")
    st.write("- 📈 **Visualizations**: Understand data trends and relationships with dynamic charts.")
    st.write("- 🔮 **Prediction**: Predict energy consumption based on multiple factors.")
    st.write("- 📚 **Conclusion**: Summarize findings and key takeaways.")
    st.write("Navigate through the sidebar to explore each section.")

# --- Summary & Insights ---
if page == "📊 Summary & Insights":
    st.header("📊 Summary & Insights")

    # Summary statistics
    st.subheader("1. 📑 Summary Statistics")
    st.write(data.describe())

    # Correlation matrix
    st.subheader("2. 🔗 Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    numeric_data = numeric_data.fillna(0)
    corr_matrix = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Top Charger Types
    st.subheader("3. 🔌 Top Charger Types by Usage")
    charger_counts = data['Charger Type'].value_counts()
    st.bar_chart(charger_counts)

# --- Visualizations ---
if page == "📈 Visualizations":
    st.header("📈 Detailed Visualizations")

    # Charging Cost Distribution
    st.subheader("1. 💰 Distribution of Charging Cost")
    bins = st.slider("Select number of bins for Charging Cost Histogram", min_value=10, max_value=50, value=20, step=5)
    fig, ax = plt.subplots()
    ax.hist(data['Charging Cost (USD)'], bins=bins, color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Charging Cost")
    ax.set_xlabel("Charging Cost (USD)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Energy Consumed vs Charging Duration
    st.subheader("2. ⚙️ Energy Consumed vs Charging Duration")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='Charging Duration (hours)', y='Energy Consumed (kWh)', hue='Charger Type', palette='viridis', ax=ax)
    ax.set_title("Energy Consumed vs Charging Duration")
    ax.set_xlabel("Charging Duration (hours)")
    ax.set_ylabel("Energy Consumed (kWh)")
    st.pyplot(fig)

    # Boxplot of Charging Rate by Charger Type
    st.subheader("3. 📦 Boxplot of Charging Rate by Charger Type")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data, x='Charger Type', y='Charging Rate (kW)', ax=ax)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Time of Day Analysis
    st.subheader("4. ⏰ Charging Sessions by Time of Day")
    time_of_day_counts = data['Time of Day'].value_counts()
    st.bar_chart(time_of_day_counts)

# --- Prediction ---
if page == "🔮 Prediction":
    st.header("🔮 Predicting Energy Consumed")

    # Preparing data for prediction
    features = ['Charging Duration (hours)', 'Charging Rate (kW)', 'Temperature (°C)', 'Vehicle Age (years)']
    target = 'Energy Consumed (kWh)'

    # Handle missing target values
    if data['Energy Consumed (kWh)'].isnull().any():
        st.warning("Missing values found in the target column 'Energy Consumed (kWh)'. Filling with the column mean.")
        data['Energy Consumed (kWh)'] = data['Energy Consumed (kWh)'].fillna(data['Energy Consumed (kWh)'].mean())

    # Check if required columns are in the dataset
    missing_features = [col for col in features if col not in data.columns]
    if missing_features:
        st.warning(f"Missing columns for prediction: {', '.join(missing_features)}")
    elif target not in data.columns:
        st.warning(f"Target column '{target}' not found in the dataset.")
    else:
        X = data[features].fillna(0)  # Ensure features have no missing values
        y = data['Energy Consumed (kWh)']

        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")

        # User input for prediction
        st.subheader("🧠 Make a Prediction")
        input_data = {feature: st.slider(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean())) for feature in features}
        input_df = pd.DataFrame([input_data])

        if st.button("Predict Energy Consumed"):
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Energy Consumed: {prediction:.2f} kWh")

# Conclusion
if page == "📚 Conclusion":
    st.header("📚 Conclusion")
    st.write("Thank you for exploring Electric Vehicle Charging Patterns. We hope these insights and predictions help in better understanding EV charging behaviors.")
