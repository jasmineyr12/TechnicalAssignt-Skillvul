
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# memasukkan dataset
url_dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
data = pd.read_csv(url_dataset)

print(data.head())
print(data.info())
print(data.describe())

data = data.drop(columns=['UDI', 'Product ID'])
kategori = data.select_dtypes(include=['object']).columns

# proses
label_encorder= {}
for col in kategori:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col]) 
    label_encorder[col] = le

# memisahan fitur dan target
K = data.drop(columns=['TWF'])
L = data['TWF']

X_train, X_test, y_train, y_test = train_test_split(K, L, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X3_train_scaled = scaler.fit_transform(X_train)
X4_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# uji coba
for name, model in models.items():
    model.fit(X3_train_scaled, y_train)

adore = {}
for name, model in models.items():
    y_pred = model.predict(X4_scaled)
    adore[name] = {
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

# hasil
hasil = pd.DataFrame(adore).T
print(hasil)

plt.figure(figsize=(8, 4))
sb.heatmap(data.corr(), annot=True, cmap='cool')
plt.xlabel('Index Sempel')
plt.ylabel('TWF')
plt.title('Heatmap of Feature Correlations')
plt.show()