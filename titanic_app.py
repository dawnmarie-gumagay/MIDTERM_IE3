import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Titanic Data Exploration App", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #E0F7FA;  
        color: #2F4F4F;  
    }
    .st-button {
        background-color: #001f3f;  
        color: #FFFFFF;
    }
    .stButton>button {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

file_path = "https://raw.githubusercontent.com/dawnmarie-gumagay/MIDTERM_IE3/main/titanic_dataset.csv"
titanic_dataset = pd.read_csv(file_path)

st.title("Titanic Data Exploration App")

st.header("Introduction")
st.write("""
This application explores the factors affecting survival rates on the Titanic. 
The dataset used for this analysis is the Titanic dataset from Kaggle. 
The purpose of this exploration is to uncover insights into the factors that influenced passengers' survival on the Titanic, 
such as their class, gender, age, and fare paid.
""")

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

st.header("Filter the Data")

age_filter = st.slider("Select Age Range:", int(titanic_dataset['Age'].min()), int(titanic_dataset['Age'].max()), (10, 50))
filtered_data = titanic_dataset[(titanic_dataset['Age'] >= age_filter[0]) & (titanic_dataset['Age'] <= age_filter[1])]

pclass_filter = st.multiselect("Select Passenger Class (1st, 2nd, 3rd):", [1, 2, 3], [1, 2, 3])
filtered_data = filtered_data[filtered_data['Pclass'].isin(pclass_filter)]

st.write("Filtered Data")
st.dataframe(filtered_data)

st.header("Visualizations")

visualization_type = st.selectbox("Select Visualization Type:", 
                                    ["Survival Rates by Class", "Survival Rates by Gender", 
                                     "Percentage of Passengers by Class", "Survival Rate Percentage", 
                                     "Correlation Heatmap", "Age Distribution", "Fare Distribution"])

if visualization_type == "Survival Rates by Class":
    st.subheader("Survival Rates by Class")
    class_survival_rate = filtered_data.groupby("Pclass")["Survived"].mean()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=class_survival_rate.index, y=class_survival_rate.values, ax=ax1, color='#001f3f')
    ax1.set_title("Survival Rate by Class", color='#001f3f')
    ax1.set_ylabel("Survival Rate", color='#001f3f')
    ax1.set_xlabel("Passenger Class", color='#001f3f')
    st.pyplot(fig1)
    st.write("This bar chart shows the survival rate of passengers by class.")

elif visualization_type == "Survival Rates by Gender":
    st.subheader("Survival Rates by Gender")
    gender_survival_rate = filtered_data.groupby("Sex")["Survived"].mean()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=gender_survival_rate.index, y=gender_survival_rate.values, ax=ax2, color='#001f3f')
    ax2.set_title("Survival Rate by Gender", color='#001f3f')
    ax2.set_ylabel("Survival Rate", color='#001f3f')
    ax2.set_xlabel("Gender", color='#001f3f')
    st.pyplot(fig2)
    st.write("This bar chart illustrates the survival rates based on gender.")

elif visualization_type == "Percentage of Passengers by Class":
    st.subheader("Percentage of Passengers by Class")
    class_counts = filtered_data['Pclass'].value_counts()
    fig3 = px.pie(filtered_data, names='Pclass', title="Percentage of Passengers by Class", hole=0.3, 
                   color_discrete_sequence=['#001f3f', '#FF6F61', '#FF6F61'])
    st.plotly_chart(fig3)
    st.write("This pie chart shows the distribution of passengers across different classes.")

elif visualization_type == "Survival Rate Percentage":
    st.subheader("Survival Rate Percentage")
    survival_counts = filtered_data['Survived'].value_counts()
    fig4 = px.pie(filtered_data, names='Survived', title="Survival Rate Percentage", hole=0.3, 
                   color_discrete_sequence=['#001f3f', '#FF6F61'])
    st.plotly_chart(fig4)
    st.write("This pie chart illustrates the overall survival rate among passengers.")

elif visualization_type == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    corr_matrix = filtered_data[['Age', 'Fare', 'SibSp', 'Parch']].corr()
    fig5, ax5 = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", ax=ax5)
    ax5.set_title("Correlation Heatmap of Features", color='#001f3f')
    st.pyplot(fig5)
    st.write("This heatmap displays the correlation between numerical features in the dataset.")

elif visualization_type == "Age Distribution":
    st.subheader("Age Distribution")
    fig6, ax6 = plt.subplots()
    sns.histplot(filtered_data['Age'], bins=20, kde=True, ax=ax6, color='#001f3f')
    ax6.set_title("Age Distribution", color='#001f3f')
    ax6.set_xlabel("Age", color='#001f3f')
    st.pyplot(fig6)
    st.write("This histogram shows the age distribution of the passengers.")

elif visualization_type == "Fare Distribution":
    st.subheader("Fare Distribution")
    fig7, ax7 = plt.subplots()
    sns.histplot(filtered_data['Fare'], bins=20, kde=True, ax=ax7, color='#001f3f')
    ax7.set_title("Fare Distribution", color='#001f3f')
    ax7.set_xlabel("Fare", color='#001f3f')
    st.pyplot(fig7)
    st.write("This histogram illustrates the fare distribution among passengers.")

st.header("Conclusion")
st.write("""
From the analysis of the Titanic dataset, several factors stood out as having a significant impact on survival:
- **Class**: Passengers in the first class had a higher survival rate compared to those in the third class.
- **Gender**: Females were more likely to survive than males.
- **Age**: Younger passengers had higher survival rates.
- **Fare**: Passengers who paid higher fares tended to have better survival rates.
These findings align with the historical accounts of the Titanic tragedy, where women, children, and the wealthy were given priority in lifeboats.
""")
