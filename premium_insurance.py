import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Medical Premium Price Model Training...\n")

# ===============================
# Step 1: Load Dataset
# ===============================
print("📂 Loading dataset...")
df = pd.read_csv('Medicalpremium.csv')
print("✅ Dataset loaded successfully!")
print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
print("\n🔹 First 5 rows:")
print(df.head(), "\n")

# ===============================
# Step 2: Check basic info
# ===============================
print("🧮 Checking for missing values...")
print(df.isnull().sum(), "\n")

# ===============================
# Step 3: Feature Engineering - BMI Calculation
# ===============================
print("⚙️ Creating BMI feature...")
df['BMI'] = df.Weight.values / (((df.Height).values / 100) ** 2)

under_index = df[df.BMI < 18.4999].index
normal_index = df[(df.BMI > 18.5) & (df.BMI < 24.9999)].index
over_index = df[(df.BMI > 25) & (df.BMI < 29.9999)].index
obecity_index = df[df.BMI > 30].index

df.loc[under_index, 'BMI_Status'] = 'Under Weight'
df.loc[normal_index, 'BMI_Status'] = 'Normal'
df.loc[over_index, 'BMI_Status'] = 'Over Weight'
df.loc[obecity_index, 'BMI_Status'] = 'Obecity'
print("✅ BMI and BMI_Status columns created.\n")

# ===============================
# Step 4: Data Cleanup
# ===============================
print("🧹 Cleaning data and creating dummy variables...")
df = df.drop(['PremiumLabel', 'AgeLabel', 'WeightLabel', 'HeightLabel'], axis=1, errors='ignore')
df_BMI_Status = pd.get_dummies(df.BMI_Status)
df = pd.concat([df, df_BMI_Status], axis=1)
df = df.drop(['BMI_Status', 'BMI'], axis=1)
print("✅ Data cleaned successfully!")
print(f"Final columns: {list(df.columns)}\n")

# ===============================
# Step 5: Train/Test Split
# ===============================
from sklearn.model_selection import train_test_split
X = df.drop('PremiumPrice', axis=1)
y = df['PremiumPrice']
print("✂️ Splitting dataset into train/test sets...")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"✅ Training set: {x_train.shape}, Testing set: {x_test.shape}\n")

# ===============================
# Step 6: Model Training (Random Forest)
# ===============================
print("🌲 Training Random Forest model...")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

classifier = RandomForestClassifier(n_estimators=37, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)
print("✅ Model training complete.\n")

# ===============================
# Step 7: Evaluate Model
# ===============================
print("📊 Evaluating model...")
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%\n")

# ===============================
# Step 8: Feature Importance
# ===============================
print("📈 Plotting feature importance...")
feature_imp = classifier.feature_importances_
sns.barplot(x=feature_imp, y=X.columns)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Importance from Random Forest")
plt.show()
plt.close('all')

# ===============================
# Step 9: Save Model
# ===============================
import pickle
print("💾 Saving trained model as 'premium_model.pkl'...")
with open('premium_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
print("✅ Model saved successfully as premium_model.pkl\n")

print("🎉 All steps completed successfully! Your model is ready for deployment 🚀")
