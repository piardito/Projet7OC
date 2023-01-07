import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
import streamlit.components.v1 as components
import plotly.graph_objects as go


st.set_page_config(layout="wide")

st.title('Credit Scoring')

st.markdown(" :money_with_wings: ",
            unsafe_allow_html=True)

@st.cache(allow_output_mutation=True,persist=True)
def load_data():
  df = pd.read_csv(r"appl_test_tr.csv")
  return df

df = load_data()



X = df.drop(columns=["SK_ID_CURR"])

modele = load(r"modele.joblib")

feature_importances=modele.feature_importances_
attributes=list(X.columns)

feat=pd.DataFrame(feature_importances)
att=pd.DataFrame(attributes)
feat_att=pd.concat([feat,att],axis=1)
feat_att.columns=['valeur','attributes']
feat_att=feat_att.sort_values(by="valeur",ascending=False)
feat_att_1=feat_att[:10]

proba=modele.predict_proba(X)[:,1]
seuil=st.slider("Choisissez le seuil",0.1,0.5,0.1)
TARGET= np.where(proba > seuil, 1, 0)
prediction= TARGET

df1 = pd.DataFrame(proba,columns=['Proba'])
df2 = pd.DataFrame(prediction,columns=['TARGET'])

df3 = pd.concat([df1,df2,df["SK_ID_CURR"]],axis=1)

Clients=st.sidebar.selectbox("Choisissez le client",df3['SK_ID_CURR'])

st.write("Client numéro : ", Clients)

conditionlist = [
    df3["TARGET"]==0,
    df3["TARGET"]==1
    ]
choicelist = ["Solvable","Non Solvable"]
df3['Décision'] = np.select(conditionlist, choicelist)

c1,c2=st.columns(2)
with c1:
    st.write("Décision")
    st.dataframe(df3[df3["SK_ID_CURR"]==Clients][['Décision']].style.background_gradient(cmap='Reds'))
with c2:
    st.write("Probabilité")
    st.dataframe((df3[df3["SK_ID_CURR"]==Clients][['Proba']]).style.background_gradient(cmap='Reds'))



@st.cache(allow_output_mutation=True,persist=True)
def load_data_1():
   data= pd.read_csv(r"appl_test.csv")
   return data
df10=load_data_1()

df10["Age"] = -(df10["DAYS_BIRTH"]/365)

df10["Ancienneté dans emploi"] = -(df10["DAYS_EMPLOYED"]/365)

df10["Montant_crédit"]=df10["AMT_CREDIT_x"]

df10["Annuités"] = df10["AMT_ANNUITY_x"]

df10["Montant_bien"] = df10['AMT_GOODS_PRICE_x']

df10["Nbre_enfants"] = df10["CNT_CHILDREN"]

df10["Total des Revenus"] = df10["AMT_INCOME_TOTAL"]

df10["Statut_familial"] = df10["NAME_FAMILY_STATUS"]

df10["Sexe"] = df10["CODE_GENDER"]

st.subheader("Informations sur le client")

st.table(df10[df10["SK_ID_CURR"]==Clients][["Sexe","Ancienneté dans emploi","Montant_crédit",
"Age","Annuités","Montant_bien","Total des Revenus"]])

st.table(df10[df10["SK_ID_CURR"]==Clients][["Statut_familial","NAME_HOUSING_TYPE","NAME_EDUCATION_TYPE",
                                                "NAME_CONTRACT_TYPE","NAME_INCOME_TYPE","Nbre_enfants"]])

st.subheader("Graphiques")

check1=st.sidebar.checkbox("Boxplot des features")

if check1:
    options = st.selectbox(
             'Choisissez votre feature',
             list(df10.columns))

    figure = px.box(df10,y=options)
    st.plotly_chart(figure,use_container_width=True)

check2=st.sidebar.checkbox("Histogramme des features")

if check2:
    options = st.selectbox(
        'Choisissez votre feature',
        list(df10.columns))

    figure1 = px.histogram(df10,x=options,nbins=25)
    st.plotly_chart(figure1,use_container_width=True)



figure4=px.bar(feat_att_1,x='valeur',y="attributes")
figure3=px.pie(df3,names="TARGET")

index = st.sidebar.selectbox("Choisissez l'index",X.index)

lime2 = LimeTabularExplainer(X,
                             feature_names=X.columns,
                             class_names=["Solvable", "Non Solvable"],
                             discretize_continuous=False)


exp1 = lime2.explain_instance(X.iloc[index],
                              modele.predict_proba,
                              num_samples=100)


col_1,col_2,col_3=st.columns(3)

with col_1:
    st.plotly_chart(figure3,use_container_width=True)

with col_2:
    st.plotly_chart(figure4,use_container_width=True)

with col_3:
    html = exp1.as_html(show_table=True)
    components.html(html, height=1000)














