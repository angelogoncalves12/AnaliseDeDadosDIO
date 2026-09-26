import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from xgboost import XGBClassifier
import shap


url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

# 1. Ver a tabela e conferir se as classes estão balanceadas
print(df.head())
print(df["Class"].value_counts(normalize=True))


# 2. Criar uma nova variável para o valor da transação
df["amount_log"] = np.log1p(df["Amount"])
print(df["amount_log"])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42
)

# A escala é aprendida no treino e aplicada no teste.
# Isso evita que dados do teste influenciem o treino.
Scaler = StandardScaler()
x_train = x_train.copy()
x_test = x_test.copy()
x_train["Amount_Scaled"] = Scaler.fit_transform(x_train[["Amount"]])
x_test["Amount_Scaled"] = Scaler.transform(x_test[["Amount"]])


# 3. Modelo inicial: regressão logística
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("\nREGRESSÃO LOGÍSTICA")
print(classification_report(y_test, y_pred))

y_probs = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr)
plt.title("ROC Curve - Regressão Logística")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
print("AUC:", roc_auc_score(y_test, y_probs))

precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.plot(recall, precision)
plt.title("Precision Recall Curve - Regressão Logística")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# 4. Random Forest
# class_weight="balanced" dá mais importância aos poucos exemplos de fraude.
random_forest = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
random_forest.fit(x_train, y_train)

# O limiar padrão é 0.50. Com 0.30, o modelo tende a marcar mais fraudes.
y_probs_rf = random_forest.predict_proba(x_test)[:, 1]
y_pred_rf = (y_probs_rf >= 0.30).astype(int)

print("\nRANDOM FOREST")
print(classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, y_probs_rf))


# 5. XGBoost dentro de Pipeline
# Pipeline organiza as etapas. Aqui há apenas o modelo, mas depois seria simples
# incluir transformação ou amostragem antes dele.
peso_fraude = (y_train == 0).sum() / (y_train == 1).sum()
xgboost = Pipeline(
    [
        (
            "model",
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=peso_fraude,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        )
    ]
)
xgboost.fit(x_train, y_train)

y_probs_xgb = xgboost.predict_proba(x_test)[:, 1]
y_pred_xgb = (y_probs_xgb >= 0.30).astype(int)

print("\nXGBOOST")
print(classification_report(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, y_probs_xgb))


# 6. Importância das variáveis na Random Forest
importancia = pd.Series(
    random_forest.feature_importances_, index=x_train.columns
).sort_values(ascending=False)
print("\nVARIÁVEIS MAIS IMPORTANTES")
print(importancia.head(10))


# 7. SHAP explica uma pequena amostra para não deixar o código lento.
# TreeExplainer funciona bem com modelos de árvores, como Random Forest.
x_shap = x_test.sample(100, random_state=42)
explainer = shap.TreeExplainer(random_forest)
shap_values = explainer.shap_values(x_shap)

# Algumas versões do SHAP devolvem uma lista; outras devolvem um array 3D.
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

shap.summary_plot(shap_values, x_shap, max_display=10)