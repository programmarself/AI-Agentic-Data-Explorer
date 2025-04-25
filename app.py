import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO

# App title with agentic framing
st.title("🤖 AI Agentic Data Explorer")

# Agent introduction
st.markdown("""
**I am your AI Data Analysis Agent.** I can autonomously:
- Analyze your dataset
- Identify key characteristics
- Recommend visualizations
- Clean and preprocess data
- Generate insights
""")

# Sidebar for agent controls
with st.sidebar:
    st.header("Agent Controls")
    
    # Agent autonomy level
    autonomy_level = st.select_slider(
        "Agent Autonomy Level",
        options=["Manual", "Guided", "Semi-Autonomous", "Fully Autonomous"],
        value="Guided"
    )
    
    # Agent goals
    analysis_goals = st.multiselect(
        "Analysis Goals",
        options=["Data Understanding", "Quality Check", "Feature Relationships", 
                "Pattern Detection", "Anomaly Detection"],
        default=["Data Understanding", "Quality Check"]
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload dataset for agent to analyze", 
                                   type=["csv", "xlsx"])

# Agent thinking state
def show_agent_thinking():
    with st.status("Agent is analyzing...", expanded=True) as status:
        st.write("🔍 Scanning data structure...")
        st.write("📊 Evaluating data quality...")
        st.write("🤔 Determining best visualizations...")
        # Simulate processing time
        time.sleep(2)
        status.update(label="Analysis complete!", state="complete")

# Main agent operation
if uploaded_file is not None:
    # Read data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Agent analysis section
    st.header("🧠 Agent Analysis Report")
    
    if autonomy_level in ["Semi-Autonomous", "Fully Autonomous"]:
        show_agent_thinking()
    
    # Automatic analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Agent Summary", 
        "Data Health", 
        "Key Insights", 
        "Recommended Visuals"
    ])
    
    with tab1:
        st.subheader("Agent's Initial Assessment")
        
        # Automatic dataset characterization
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        
        st.write("### Dataset Characteristics")
        st.write(f"- Numeric columns: {len(df.select_dtypes(include=np.number).columns)}")
        st.write(f"- Categorical columns: {len(df.select_dtypes(exclude=np.number).columns)}")
        st.write(f"- Duplicate rows: {df.duplicated().sum()}")
        
        if autonomy_level == "Fully Autonomous":
            st.write("### Agent Recommendations")
            if df.isnull().sum().sum() > 0:
                st.warning("Recommendation: Consider imputing missing values")
            if df.duplicated().sum() > 0:
                st.warning("Recommendation: Remove duplicate rows")
    
    with tab2:
        st.subheader("Data Health Report")
        
        # Automatic quality analysis
        st.write("### Missing Values Analysis")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.bar_chart(missing[missing > 0])
            st.write("Columns with missing values:")
            st.write(missing[missing > 0])
        else:
            st.success("No missing values detected")
        
        st.write("### Data Type Assessment")
        dtype_counts = df.dtypes.value_counts()
        st.write(dtype_counts)
    
    with tab3:
        st.subheader("Automatic Insights")
        
        # Generate basic insights
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            st.write("### Numeric Features Summary")
            st.write(df[numeric_cols].describe().T)
            
            # Automatic correlation insight
            if len(numeric_cols) > 1:
                st.write("### Strongest Correlations")
                corr = df[numeric_cols].corr().unstack().sort_values(ascending=False)
                # Filter out self-correlations and duplicates
                corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)]
                corr = corr[~corr.index.duplicated()]
                st.write(corr.head(3))
        
        # Categorical insights
        categorical_cols = df.select_dtypes(exclude=np.number).columns
        if len(categorical_cols) > 0:
            st.write("### Categorical Features Distribution")
            for col in categorical_cols:
                st.write(f"**{col}**")
                st.write(df[col].value_counts().head())
    
    with tab4:
        st.subheader("Recommended Visualizations")
        
        # Automatic visualization recommendations
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns
        
        if len(numeric_cols) >= 2:
            st.write("### Recommended: Scatter Plot")
            col1, col2 = st.columns(2)
            x_axis = col1.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_axis = col2.selectbox("Y-axis", numeric_cols, key="scatter_y")
            
            fig = px.scatter(df, x=x_axis, y=y_axis, 
                            hover_data=df.columns,
                            title=f"{x_axis} vs {y_axis}")
            st.plotly_chart(fig)
            
            if len(categorical_cols) > 0:
                st.write("### Recommended: Categorical Distribution")
                cat_col = st.selectbox("Select categorical column", categorical_cols)
                fig = px.histogram(df, x=cat_col, title=f"Distribution of {cat_col}")
                st.plotly_chart(fig)
        
        if len(numeric_cols) > 0:
            st.write("### Recommended: Numeric Distributions")
            num_col = st.selectbox("Select numeric column", numeric_cols)
            fig = px.box(df, y=num_col, title=f"Distribution of {num_col}")
            st.plotly_chart(fig)
    
    # Agent action section
    st.header("🛠 Agent Actions")
    
    if st.checkbox("Show data cleaning options"):
        st.write("### Data Cleaning Tools")
        
        if df.isnull().sum().sum() > 0:
            st.write("#### Missing Value Handling")
            col1, col2 = st.columns(2)
            method = col1.selectbox("Imputation method", 
                                  ["Drop rows", "Fill with mean", "Fill with median", 
                                   "Fill with mode", "Fill with zero"])
            
            if st.button("Apply missing value treatment"):
                if method == "Drop rows":
                    df = df.dropna()
                elif method == "Fill with mean":
                    df = df.fillna(df.mean())
                elif method == "Fill with median":
                    df = df.fillna(df.median())
                elif method == "Fill with mode":
                    df = df.fillna(df.mode().iloc[0])
                else:
                    df = df.fillna(0)
                st.success("Missing values treated!")
                st.experimental_rerun()
        
        if df.duplicated().sum() > 0:
            st.write("#### Duplicate Handling")
            if st.button("Remove duplicate rows"):
                df = df.drop_duplicates()
                st.success(f"Removed {df.duplicated().sum()} duplicates!")
                st.experimental_rerun()
    
    # Agent suggestions
    st.header("💡 Agent Suggestions")
    if len(numeric_cols) >= 3:
        st.write("Consider creating a pairplot to examine relationships between numeric variables")
    if len(categorical_cols) >= 2:
        st.write("Consider cross-tabulating categorical variables to examine their relationships")
    
    # Download processed data
    st.download_button(
        label="Download Processed Data",
        data=df.to_csv(index=False),
        file_name='agent_processed_data.csv',
        mime='text/csv'
    )
    
else:
    st.info("Please upload a dataset for the AI agent to analyze")
    st.markdown("""
    ### The AI Agent can analyze:
    - CSV files
    - Excel files
    - Up to 100MB in size
    
    **Sample datasets to try:**
    - [Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
    - [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
    """)