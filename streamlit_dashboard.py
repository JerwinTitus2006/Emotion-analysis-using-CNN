import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the detection log CSV
st.set_page_config(layout="wide")
st.title("🧠 Age, Gender & Emotion Detection Dashboard")

try:
    df = pd.read_csv("C:/Users/Jerwin titus/Desktop/DE/detections.csv", parse_dates=["timestamp"])

    # Sidebar filters
    st.sidebar.header("🔎 Filters")
    selected_gender = st.sidebar.multiselect("Select Gender", df["gender"].unique(), default=df["gender"].unique())
    selected_emotion = st.sidebar.multiselect("Select Emotion", df["emotion"].unique(), default=df["emotion"].unique())

    filtered_df = df[df["gender"].isin(selected_gender) & df["emotion"].isin(selected_emotion)]

    # KPIs
    st.subheader("📊 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", len(filtered_df))
    col2.metric("Most Common Gender", filtered_df['gender'].mode()[0] if not filtered_df.empty else "N/A")
    col3.metric("Most Common Emotion", filtered_df['emotion'].mode()[0] if not filtered_df.empty else "N/A")

    st.divider()

    # Charts
    st.subheader("📍 Detection Trends")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Gender Distribution")
        gender_counts = filtered_df["gender"].value_counts()
        st.bar_chart(gender_counts)

    with col2:
        st.markdown("### Emotion Distribution")
        emotion_counts = filtered_df["emotion"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    st.divider()

    st.subheader("📆 Emotion Over Time")
    df_time = filtered_df.copy()
    df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
    df_time.set_index("timestamp", inplace=True)
    emotion_time = df_time.resample("1T")["emotion"].apply(lambda x: x.value_counts().index[0] if not x.empty else None)

    st.line_chart(emotion_time.fillna(method='ffill'))

    st.divider()
    st.subheader("📝 Raw Log Data")
    st.dataframe(filtered_df)

except FileNotFoundError:
    st.warning("Log file not found. Please run detection first and generate `detection_log.csv`.")