from sklearn.datasets import make_classification
import pandas as pd

x, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,
    weights=[0.98, ], class_sep=0.5, scale=1.0, shuffle=True, flip_y=0, random_state=0
)

hourly_traffic = [
    120, 123, 124, 119, 196,
    121, 118, 117, 500, 132
]

pd.Series(hourly_traffic) > pd.Series(hourly_traffic).quantile(0.95)

class PercentileDetection:
    def __init__(self, percentile=0.9):
        self.percentile = percentile
    def fit(self, x, y=None):
        self.threshold = pd.Series(x).quantile(self.percentile)
    def predict(self, x, y=None):
        return (pd.Series(x) > self.threshold).values
    def fit_predict(self, x, y=None):
        self.fit(x)
        return self.predict(x)

outlierd = PercentileDetection(percentile=0.95)
df = pd.DataFrame(
    { 
        'hourly_traffic' : hourly_traffic,
        'is_outlier' : outlierd.fit_predict(hourly_traffic)
    }
).style.apply(
    lambda row: ['font-weight:bold'] * len(row)
      if row['is_outlier'] == True
      else ['font-weight: normal'] * len(row), axis=1 
)
print(df)