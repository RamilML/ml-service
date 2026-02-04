import pandas as pd

df = pd.read_csv("Student_performance_data _.csv")
features = [
    "Age",
    "Gender",
    "StudyTimeWeekly",
    "Absences",
    "Extracurricular"
]
target = "GPA"
X = df[features]
y = df[target]

print(df.head())
print(df.shape)
print(df.columns)