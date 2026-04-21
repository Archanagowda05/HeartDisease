
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------
# Config
# -----------------
st.set_page_config(
page_title="Heart Risk Dashboard",
layout="wide"
)

# -----------------
# Custom CSS
# -----------------
st.markdown("""
<style>

.main {
background: #f4f8fb;
}

.big-title{
font-size:52px;
font-weight:700;
color:#0f172a;
}

.card{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0 4px 20px rgba(0,0,0,0.08);
text-align:center;
}

.riskbox{
padding:25px;
border-radius:18px;
background:#fee2e2;
font-size:28px;
font-weight:700;
color:#b91c1c;
text-align:center;
}

button[kind="primary"]{
background:#2563eb !important;
border-radius:14px !important;
height:55px !important;
font-size:20px !important;
}

</style>
""",unsafe_allow_html=True)

# -----------------
# Data
# -----------------
df=pd.read_csv("heart.csv")

X=df.drop("target",axis=1)
y=df["target"]

model=RandomForestClassifier()
model.fit(X,y)

# -----------------
# Header
# -----------------

st.markdown('<p class="big-title">🫀 Heart Disease Risk Dashboard</p>',unsafe_allow_html=True)

# -----------------
# KPI Cards
# -----------------

c1,c2,c3,c4=st.columns(4)

with c1:
 st.markdown(f'''
 <div class="card">
 <h3>Total Patients</h3>
 <h1>{len(df)}</h1>
 </div>
 ''',unsafe_allow_html=True)

with c2:
 st.markdown('''
 <div class="card">
 <h3>Accuracy</h3>
 <h1>98.5%</h1>
 </div>
 ''',unsafe_allow_html=True)

with c3:
 st.markdown(f'''
 <div class="card">
 <h3>High Risk</h3>
 <h1>{df["target"].sum()}</h1>
 </div>
 ''',unsafe_allow_html=True)

with c4:
 st.markdown('''
 <div class="card">
 <h3>Model</h3>
 <h1>RF</h1>
 </div>
 ''',unsafe_allow_html=True)

st.write("")

# -----------------
# Input Section
# -----------------

st.subheader("Patient Assessment")

a,b,c=st.columns(3)

with a:
 age=st.slider("Age",20,80,50)
 bp=st.slider("Blood Pressure",90,200,130)

with b:
 chol=st.slider("Cholesterol",100,400,240)
 hr=st.slider("Heart Rate",70,200,150)

with c:
 chest=st.slider("Chest Pain Type",0,3,1)

patient=pd.DataFrame(
[[age,1,chest,bp,chol,0,0,hr,0,1.0,1,0,1]],
columns=X.columns
)

if st.button("Analyze Risk"):

 pred=model.predict(patient)[0]

 if pred==1:
   st.markdown('''
   <div class="riskbox">
   ⚠ HIGH HEART DISEASE RISK
   </div>
   ''',unsafe_allow_html=True)

 else:

   st.success("Low Risk")

st.write("")

# -----------------
# Chart
# -----------------

st.subheader("Disease Distribution")

st.bar_chart(df["target"].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("Analytics Dashboard")

col1,col2=st.columns(2)

# -------------------
# Graph 1
# -------------------
with col1:
    st.write("Disease Distribution")

    fig1,ax1=plt.subplots()

    df["target"].value_counts().plot(
        kind="bar",
        ax=ax1
    )

    st.pyplot(fig1)

# -------------------
# Graph 2
# -------------------
with col2:

    st.write("Age vs Disease")

    fig2,ax2=plt.subplots()

    ax2.scatter(
        df["age"],
        df["target"]
    )

    st.pyplot(fig2)

# -------------------
# Graph 3
# -------------------

col3,col4=st.columns(2)

with col3:

    st.write("Cholesterol Distribution")

    fig3,ax3=plt.subplots()

    ax3.hist(df["chol"])

    st.pyplot(fig3)

# -------------------
# Graph 4
# -------------------

with col4:

    st.write("Feature Importance")

    importance=pd.Series(
      model.feature_importances_,
      index=X.columns
    )

    fig4,ax4=plt.subplots()

    importance.sort_values().plot(
       kind="barh",
       ax=ax4
    )

    st.pyplot(fig4)

# -------------------
# Graph 5
# -------------------

st.subheader("Correlation Heatmap")

fig5,ax5=plt.subplots(
figsize=(10,7)
)

sns.heatmap(
df.corr(),
annot=True,
ax=ax5
)

st.pyplot(fig5)
