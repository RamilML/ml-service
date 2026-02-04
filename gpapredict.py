import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

df = pd.read_csv("Student_performance_data _.csv")
features = [
    "Gender",
    "StudyTimeWeekly",
    "Absences",
    "Extracurricular"
]
target = "GPA"
X = df[features]
y = df[target]

X_np = X.to_numpy()
y_np = y.to_numpy()

n = len(df)
rng = np.random.default_rng(42)
indices = rng.permutation(n)

test_size = 0.2
n_test = int(n * test_size)

test_idx = indices[:n_test]
train_idx = indices[n_test:]

X_train, X_test = X_np[train_idx], X_np[test_idx]
y_train, y_test = y_np[train_idx], y_np[test_idx]

print("Train size:", X_train.shape, "Test size:", X_test.shape)


#print(df.head()
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.isna().sum())
