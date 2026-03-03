import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create Dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "Hours_Slept": [8, 7, 7, 6, 6, 5, 5, 4],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 85]
}

df = pd.DataFrame(data)

# Step 2: Define Features and Target
X = df[["Hours_Studied", "Hours_Slept"]]
y = df["Marks"]

# Step 3: Train Model
model = LinearRegression()
model.fit(X, y)

# Step 4: Take User Input
study = float(input("Enter hours studied: "))
sleep = float(input("Enter hours slept: "))

prediction = model.predict([[study, sleep]])

print(f"Predicted Marks: {prediction[0]:.2f}")

# Step 5: Visualization
plt.scatter(df["Hours_Studied"], df["Marks"])
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()