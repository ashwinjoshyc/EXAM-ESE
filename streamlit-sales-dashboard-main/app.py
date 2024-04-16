import pandas as pd  
import plotly.express as px  
import streamlit as st  

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

# ---- READ EXCEL ----
@st.cache_data
def get_data_from_excel():
    df = pd.read_csv("exam.csv")
    return df

df = get_data_from_excel()

# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")

# Check if "Age", "Rating", and "Positive Feedback Count" columns exist
required_columns = ["Age", "Rating", "Positive Feedback Count"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.warning(f"One or more required columns are not found in the dataset: {', '.join(missing_columns)}")
    st.stop()

age_range = st.sidebar.slider("Select Age Range:", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=(int(df["Age"].min()), int(df["Age"].max())))
rating_range = st.sidebar.slider("Select Rating Range:", min_value=int(df["Rating"].min()), max_value=int(df["Rating"].max()), value=(int(df["Rating"].min()), int(df["Rating"].max())))
feedback_count_range = st.sidebar.slider("Select Positive Feedback Count Range:", min_value=int(df["Positive Feedback Count"].min()), max_value=int(df["Positive Feedback Count"].max()), value=(int(df["Positive Feedback Count"].min()), int(df["Positive Feedback Count"].max())))

df_selection = df.query(
    "(@age_range[0] <= Age <= @age_range[1]) & (@rating_range[0] <= Rating <= @rating_range[1]) & (@feedback_count_range[0] <= `Positive Feedback Count` <= @feedback_count_range[1])"
)

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop() # This will halt the app from further execution.

# ---- MAINPAGE ----
st.title(":bar_chart: Sales Dashboard")
st.markdown("##")

# 3D SCATTER PLOT
fig_3d_scatter = px.scatter_3d(
    df_selection,
    x="Age",
    y="Rating",
    z="Positive Feedback Count",
    title="<b>3D Scatter Plot: Age, Rating, and Positive Feedback Count</b>",
    color="Rating",  # Using "Rating" for color
    template="plotly_white",
)
fig_3d_scatter.update_layout(
    scene=dict(
        xaxis_title="Age",
        yaxis_title="Rating",
        zaxis_title="Positive Feedback Count"
    ),
)

st.plotly_chart(fig_3d_scatter, use_container_width=True)
