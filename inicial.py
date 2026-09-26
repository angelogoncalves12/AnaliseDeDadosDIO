import pandas as pd
import numpy as np 
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

print (df.head())

#ler se tá balanceado
print (df["Class"].value_counts(normalize=True))

#
df["amount_log"] = np.log1p(df["Amount"])
print (df["amount_log"])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42
)

Scaler = StandardScaler()
df["Amount_Scaled"] = Scaler.fit_transform(df["Amount"])

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print (classification_report(y_test, y_pred))

y_probs = model.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot (fpr, tpr)
plt.title ("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
print ("AUC: ", roc_auc_score(y_test, y_probs))

precision,recall, _ = precision_recall_curve(y_test, y_probs)
plt.plot (recall, precision)
plt.title ("Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel ("Precision")
plt.show()