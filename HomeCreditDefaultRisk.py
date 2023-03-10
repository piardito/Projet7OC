import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
import streamlit.components.v1 as components
import plotly.graph_objects as go
import requests
import json
import shap
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout="wide")

st.title('Credit Scoring')

st.markdown(" :money_with_wings: ",
            unsafe_allow_html=True)


def score(sk_id):
    g = requests.get("https://projet7ocapi.herokuapp.com/" +
                     "score/?SK_ID_CURR=" + str(sk_id))
    resultat = json.loads(g.content)
    df_api = pd.DataFrame(resultat.items()).set_index(0).T
    df_api.set_index("SK_ID_CURR", inplace=True)
    return(df_api.loc[str(sk_id)])


@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv(r"appl_test_tr.csv")
    return df


df = load_data()

modele = load(r"mod.joblib")


feature_importances = modele.feature_importances_
attributes = list(df.set_index("SK_ID_CURR").columns)
feat = pd.DataFrame(feature_importances)
att = pd.DataFrame(attributes)
feat_att = pd.concat([feat, att], axis=1)
feat_att.columns = ['valeur', 'attributes']
feat_att = feat_att.sort_values(by="valeur", ascending=False)
feat_att_1 = feat_att[:10]

proba = modele.predict_proba(df.set_index("SK_ID_CURR"))[:, 1]
seuil = st.slider("Choisissez le seuil", 0.1, 0.5, 0.1)
TARGET = np.where(proba > seuil, 1, 0)
prediction = TARGET

df1 = pd.DataFrame(proba, columns=['Proba'])
df2 = pd.DataFrame(prediction, columns=['TARGET'])
df3 = pd.concat([df1, df2, df["SK_ID_CURR"]], axis=1)


Clients = st.sidebar.selectbox("Choisissez le client", df3["SK_ID_CURR"])


conditionlist = [
    df3["TARGET"] == 0,
    df3["TARGET"] == 1
]
choicelist = ["Solvable", "Non Solvable"]
df3['Décision'] = np.select(conditionlist, choicelist)

df3["Score"] = 100 - 100 * df3["Proba"]


c1, c2 = st.columns(2)
with c1:
    st.write("Décision")
    st.table(df3[df3["SK_ID_CURR"] == Clients][['Décision']].style.set_properties(
        **{'background-color': 'blue', 'color': 'yellow'}))


with c2:
    st.write("Scoring")
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        #value= int(np.rint(df3[df3["SK_ID_CURR"]==Clients]['Score'])),
        value=float(score(Clients)),
        mode="gauge+number+delta",
        title={'text': f"Score du client {Clients}"},
        delta={'reference': 100*(1-seuil)},
        gauge={'axis': {'range': [None, 100]},
               'steps': [
                   {'range': [0, 100*(1-seuil)], 'color': "lightgray"},
                   {'range': [100*(1-seuil), 100], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100*(1-seuil)}}))
    st.plotly_chart(fig, use_container_width=False)


@st.cache(allow_output_mutation=True)
def load_data_1():
    data = pd.read_csv(r"appl_test.csv")
    return data


df10 = load_data_1()


df10["Age"] = -(df10["DAYS_BIRTH"]/365)

df10["Ancienneté dans emploi"] = -(df10["DAYS_EMPLOYED"]/365)

df10["Montant_crédit"] = df10["AMT_CREDIT_x"]

df10["Annuités"] = df10["AMT_ANNUITY_x"]

df10["Montant_bien"] = df10['AMT_GOODS_PRICE_x']

df10["Nbre_enfants"] = df10["CNT_CHILDREN"]

df10["Total des Revenus"] = df10["AMT_INCOME_TOTAL"]

df10["Statut_familial"] = df10["NAME_FAMILY_STATUS"]

df10["Sexe"] = df10["CODE_GENDER"]

st.subheader("Informations sur le client")

st.table(df10[df10["SK_ID_CURR"] == Clients][["Sexe", "Ancienneté dans emploi", "Montant_crédit", "Age",
                                              "Annuités", "Montant_bien", "Total des Revenus"]].style.set_properties(**{'background-color': 'cyan', 'color': 'black'}))

st.table(df10[df10["SK_ID_CURR"] == Clients][["Statut_familial", "NAME_HOUSING_TYPE", "NAME_EDUCATION_TYPE",
                                              "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "Nbre_enfants"]].style.set_properties(**{'background-color': 'cyan', 'color': 'black'}))

st.subheader("Graphiques")

check1 = st.sidebar.checkbox("Boxplot des features")

if check1:
    options = st.selectbox(
        'Choisissez votre feature',
        list(df10.columns), key="histogramme")

    figure = px.box(df10, y=options)
    st.plotly_chart(figure, use_container_width=True)

check2 = st.sidebar.checkbox("Histogramme des features")

if check2:
    options1 = st.selectbox(
        'Choisissez votre feature',
        list(df10.columns), key="boxplot")

    figure1 = px.histogram(df10, x=options1, nbins=25)
    st.plotly_chart(figure1, use_container_width=True)

check3 = st.sidebar.checkbox("Analyse bivariée")

if check3:
    options2 = st.selectbox(
        'Choisissez votre feature',
        list(df10.columns))
    options3 = st.selectbox(
        'Choisissez votre feature',
        list(df10[["Ancienneté dans emploi", "Montant_crédit",
                   "Age", "Annuités", "Montant_bien", "Total des Revenus"]].columns), key="analyse bivariée")

    figure2 = px.scatter(df10, x=options2, y=options3)
    st.plotly_chart(figure2, theme="streamlit", use_container_width=True)

figure4 = px.bar(feat_att_1, x='valeur', y="attributes")
figure4.update_layout(title_text='Features importances au niveau globale')

figure3 = px.pie(df3, names="Décision")
figure3.update_layout(title="Répartition des classes")


# lime2 = LimeTabularExplainer(df.set_index("SK_ID_CURR"),
# feature_names=df.set_index("SK_ID_CURR").columns,
#class_names=["Solvable", "Non Solvable"],
# discretize_continuous=False)


# exp1 = lime2.explain_instance(df.set_index("SK_ID_CURR").loc[Clients],
# modele.predict_proba,
# num_samples=100)

train_data = shap.sample(df.set_index("SK_ID_CURR"), random_state=0)

explainer = shap.KernelExplainer(modele.predict_proba, train_data)

fig = plt.figure(figsize=(6.2, 12))
plt.title(f"Explication locale pour la classe solvable du client {Clients}", size=20,fontweight="bold")
fig = shap.bar_plot(explainer.shap_values(df.set_index("SK_ID_CURR").loc[Clients], l1_reg="aic")[0],
                    feature_names=df.set_index("SK_ID_CURR").columns, max_display=10)

col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.plotly_chart(figure3, use_container_width=True)

with col_2:
    st.plotly_chart(figure4, use_container_width=True)

# with col_3:
    #html = exp1.as_html(show_table=True)
    #components.html(html, height=1000)
with col_3:
    st.pyplot(fig)










