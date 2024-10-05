import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import base64

# Set page configuration
st.set_page_config(page_title="Titanic Data Exploration App", layout="wide")

# CSS styling for the app
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #E0F7FA;  
        color: #2F4F4F;  
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #001f3f;  
        color: #FFFFFF;
        border-radius: 5px;
        padding: 10px 20px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #003366;
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        color: #001f3f;
    }
    .header {
        color: #2E8B57;
        font-weight: bold;
    }
    .subheader {
        color: #4B0082;
    }
    .plot-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the Titanic dataset
file_path = "https://raw.githubusercontent.com/dawnmarie-gumagay/MIDTERM_IE3/main/titanic_dataset.csv"
titanic_dataset = pd.read_csv(file_path)

# App title
st.markdown('<h1 class="title">Titanic Data Exploration App</h1>', unsafe_allow_html=True)

# Define pages for navigation
pages = {
    "Introduction": "intro",
    "Data Visualization": "visualization",
    "Conclusion": "conclusion"
}

# Create the sidebar for navigation
selected_page = st.sidebar.radio("Select a page:", list(pages.keys()))

# Introduction Page
if selected_page == "Introduction":
    # Introduction Section
    st.header("Introduction")
    st.write(""" 
    This application explores the factors affecting survival rates on the Titanic. 
    The dataset used for this analysis is the Titanic dataset from Kaggle. 
    The purpose of this exploration is to uncover insights into the factors that influenced passengers' survival on the Titanic, 
    such as their class, gender, age, and fare paid.
    """)

    # Display basic dataset information
    st.subheader("Dataset Overview")
    st.write(titanic_dataset.describe())  # Shows descriptive statistics of the dataset

    # Display a preview of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(titanic_dataset.head())  # Shows the first few rows of the dataset

    # Group Information
    st.subheader("Group Information")
    st.write(""" 
    **Group Name:** AJA  
    **Leader:** Jerry Doriquez  
    **Members:**
    - Gumagay, Dawn Marie D.
    - Ursal, Gelu Marie
    - Corbete, Areej Charisse
    - Paner, Leemar
    """)

# Data Visualization Page
elif selected_page == "Data Visualization":
    # Filter Data Section
    st.header("Filter the Data")

    age_filter = st.slider("Select Age Range:", int(titanic_dataset['Age'].min()), int(titanic_dataset['Age'].max()), (10, 50))
    filtered_data = titanic_dataset[(titanic_dataset['Age'] >= age_filter[0]) & (titanic_dataset['Age'] <= age_filter[1])]

    pclass_filter = st.multiselect("Select Passenger Class (1st, 2nd, 3rd):", [1, 2, 3], [1, 2, 3])
    filtered_data = filtered_data[filtered_data['Pclass'].isin(pclass_filter)]

    st.write("Filtered Data")
    st.dataframe(filtered_data)

    # Visualizations Section
    st.header("Visualizations")

    visualization_type = st.selectbox("Select Visualization Type:", 
                                        ["Survival Rates by Class", "Survival Rates by Gender", 
                                         "Percentage of Passengers by Class", "Survival Rate Percentage", 
                                         "Correlation Heatmap", "Age Distribution", "Fare Distribution"])

    def plot_survival_by_class():
        plt.figure(figsize=(10, 5))
        sns.set_theme(style="whitegrid")  # Change to whitegrid theme
        class_survival_rate = filtered_data.groupby("Pclass")["Survived"].mean()
        sns.barplot(x=class_survival_rate.index, y=class_survival_rate.values, palette='Blues')
        plt.title("Survival Rate by Class", fontsize=16, fontweight='bold')
        plt.ylabel("Survival Rate", fontsize=12)
        plt.xlabel("Passenger Class", fontsize=12)
        plt.xticks(rotation=0)
        st.pyplot(plt)

    def plot_survival_by_gender():
        plt.figure(figsize=(10, 5))
        sns.set_theme(style="whitegrid")
        gender_survival_rate = filtered_data.groupby("Sex")["Survived"].mean()
        sns.barplot(x=gender_survival_rate.index, y=gender_survival_rate.values, palette='pastel')
        plt.title("Survival Rate by Gender", fontsize=16, fontweight='bold')
        plt.ylabel("Survival Rate", fontsize=12)
        plt.xlabel("Gender", fontsize=12)
        plt.xticks(rotation=0)
        st.pyplot(plt)

    def plot_class_distribution():
        fig = px.pie(filtered_data, names='Pclass', title="Percentage of Passengers by Class", hole=0.3, 
                     color_discrete_sequence=px.colors.sequential.Blues)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(width=600, height=400, title_font=dict(size=16, family="Arial"))
        st.plotly_chart(fig)

    def plot_survival_rate():
        fig = px.pie(filtered_data, names='Survived', title="Survival Rate Percentage", hole=0.3, 
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(width=600, height=400, title_font=dict(size=16, family="Arial"))
        st.plotly_chart(fig)

    def plot_correlation_heatmap():
        corr_matrix = filtered_data[['Age', 'Fare', 'SibSp', 'Parch']].corr()
        plt.figure(figsize=(10, 5))
        sns.set_theme(style="white")
        sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap of Features", fontsize=16, fontweight='bold')
        st.pyplot(plt)

    def plot_age_distribution():
        plt.figure(figsize=(10, 5))
        sns.set_theme(style="whitegrid")
        sns.histplot(filtered_data['Age'], bins=20, kde=True, color='#001f3f', alpha=0.6)
        plt.title("Age Distribution", fontsize=16, fontweight='bold')
        plt.xlabel("Age", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        st.pyplot(plt)

    def plot_fare_distribution():
        plt.figure(figsize=(10, 5))
        sns.set_theme(style="whitegrid")
        sns.histplot(filtered_data['Fare'], bins=20, kde=True, color='#001f3f', alpha=0.6)
        plt.title("Fare Distribution", fontsize=16, fontweight='bold')
        plt.xlabel("Fare", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        st.pyplot(plt)

    # Plot based on user's selection
    if visualization_type == "Survival Rates by Class":
        st.subheader("Survival Rates by Class")
        plot_survival_by_class()
        st.write("First-class passengers had significantly higher survival rates compared to third-class passengers.")

    elif visualization_type == "Survival Rates by Gender":
        st.subheader("Survival Rates by Gender")
        plot_survival_by_gender()
        st.write("Females had a significantly higher chance of survival than males.")

    elif visualization_type == "Percentage of Passengers by Class":
        st.subheader("Percentage of Passengers by Class")
        plot_class_distribution()
        st.write("The majority of passengers were in third class.")

    elif visualization_type == "Survival Rate Percentage":
        st.subheader("Survival Rate Percentage")
        plot_survival_rate()
        st.write("Overall, a minority of passengers survived.")

    elif visualization_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        plot_correlation_heatmap()
        st.write("Fare and Age have weak correlations with survival, while family-related variables show a mild correlation.")

    elif visualization_type == "Age Distribution":
        st.subheader("Age Distribution")
        plot_age_distribution()
        st.write("The majority of passengers were aged between 20 and 40.")

    elif visualization_type == "Fare Distribution":
        st.subheader("Fare Distribution")
        plot_fare_distribution()
        st.write("Fares varied significantly among passengers, with a few individuals paying much higher fares.")

# Conclusion Page
elif selected_page == "Conclusion":
    st.header("Conclusion")
    st.write(""" 
    This exploration of the Titanic dataset revealed key insights regarding the factors influencing survival rates among passengers. 
    It was found that passenger class, gender, and age played significant roles in determining survival chances. 
    Higher class passengers and females had notably higher survival rates, highlighting socio-economic factors affecting survival. 
    Additionally, the data illustrated variations in age and fare among the passengers, contributing to understanding the demographics aboard the Titanic. 
    The findings emphasize the importance of socio-economic status and gender during disasters, relevant to both historical and modern contexts.
    """)
