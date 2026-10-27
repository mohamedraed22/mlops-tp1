import numpy as np
import pandas as pd


np.random.seed(42)

n_customers = 500

df = pd.DataFrame(
    {
        "age": np.random.randint(18, 70, size=n_customers),
        "salaire": np.random.randint(1200, 8000, size=n_customers),
        "abonnement": np.random.choice(
            ["Basique", "Standard", "Premium"], size=n_customers
        ),
        "anciennete": np.random.randint(0, 15, size=n_customers),
        "consommation": np.random.randint(0, 100, size=n_customers),
    }
)

df["churn"] = ((df["consommation"] < 30) & (df["anciennete"] < 2)).astype(int)

df.to_csv("churn_data.csv", index=False)

print(df.head())
print(f"Taux de churn : {df['churn'].mean() * 100:.1f}%")