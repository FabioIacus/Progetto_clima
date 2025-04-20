import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Carica il file CSV saltando la prima riga di intestazione doppia
df = pd.read_csv("GLB.Ts+dSST.csv", skiprows=1)

# Rimuovi eventuali spazi vuoti nelle intestazioni
df.columns = df.columns.str.strip()

# Sostituisci valori "***" con NaN (Not a Number)
df = df.replace("***", pd.NA)

# Converti le colonne numeriche in float
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["J-D"] = pd.to_numeric(df["J-D"], errors="coerce")  # Media annuale

# Rimuovi le righe con valori mancanti in "J-D"
df = df.dropna(subset=["J-D"])

# Regressione su tutto il periodo (1880-oggi)
# Prepara i dati per la regressione
X = df["Year"].values.reshape(-1, 1)    # Anni (feature)
y = df["J-D"].values.reshape(-1, 1)     # Temperature (target)
# Crea e addestra il modello
model = LinearRegression()
model.fit(X, y)
# Calcola la retta di regressione
y_pred = model.predict(X)

# Regressione ultimi 30 anni
df_recent = df[df["Year"] >= df["Year"].max() - 30]
X_recent = df_recent["Year"].values.reshape(-1, 1)
y_recent = df_recent["J-D"].values.reshape(-1, 1)
model_recent = LinearRegression()
model_recent.fit(X_recent, y_recent)
y_pred_recent = model_recent.predict(X_recent)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["J-D"], color='blue', label='Anomalia temperatura (1880-oggi)')
plt.plot(df["Year"], y_pred, color='red', label='Trend lineare completo')
plt.plot(df_recent["Year"], y_pred_recent, color='orange', linestyle='--', label='Trend ultimi 30 anni')

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Anno")
plt.ylabel("Anomalia temperatura (°C)")
plt.title("Trend della temperatura globale")
plt.legend()
plt.grid(True)

# Stampa del coefficiente della regressione
print(f"Innalzamento medio annuo complessivo: {model.coef_[0][0]:.4f} °C/anno")
print(f"Innalzamento medio annuo ultimi 30 anni: {model_recent.coef_[0][0]:.4f} °C/anno")

# Visualizza il grafico
plt.tight_layout()
# Salva il grafico
plt.savefig("trend_temperatura.png", dpi=300)

plt.show()
