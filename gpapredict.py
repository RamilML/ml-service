import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

df = pd.read_csv("Student_performance_data _.csv")
features = [
    "Gender",
    "StudyTimeWeekly",
    "Absences",
    "Extracurricular",
    "ParentalEducation"
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
w = np.array([0,0,0,0,0,0])
print("Train size:", X_train.shape, "Test size:", X_test.shape)

ones_train = np.ones((X_train.shape[0], 1))
ones_test = np.ones((X_test.shape[0], 1))
x_train = np.hstack((ones_train, X_train))
x_test = np.hstack((ones_test, X_test))
lm = 0.01
nm = 0.0003
def model(w, x):
    return x @ w 
def loss(y, x, w):
    return np.mean((y - model(w,x))**2)
def Dloss(y,x,w):
    grad = (2/n) * (x.T @ (model(w,x) - y))
    grad[1:] += 2 * lm * w[1:]
    return grad

for _ in range(200000):
    w = w - nm*Dloss(y_train,x_train,w)
print (w)

arr = list(map(int,input().split()))
np_ar = np.array(arr)
print(model(w,np_ar))
# feature_idx = features.index("ParentalEducation") + 1  # +1 из-за bias
# x_line = np.zeros((100, x_train.shape[1]))
# x_line[:, 0] = 1  # bias
# x_line[:, feature_idx] = np.linspace(
#     X_train[:, feature_idx - 1].min(),
#     X_train[:, feature_idx - 1].max(),
#     100
# )

# # остальные признаки = средние
# for j in range(1, x_train.shape[1]):
#     if j != feature_idx:
#         x_line[:, j] = X_train[:, j - 1].mean()

# y_line = model(w, x_line)
# plt.scatter(
#     X_train[:, feature_idx - 1],
#     y_train,
#     alpha=0.3,
#     label="Train data"
# )
# plt.plot(
#     x_line[:, feature_idx],
#     y_line,
#     color="red",
#     label="Model (slice)"
# )
# plt.xlabel("ParentalEducation")
# plt.ylabel("GPA")
# plt.legend()
# plt.show()