import pandas as pd
import numpy as np

df = pd.read_csv("data/Ratings_Warriner_et_al.csv")
sds = []
for col_name in ['V.SD.Sum', 'A.SD.Sum', 'D.SD.Sum']:
    sds = sds + list(df[col_name])

print(np.sqrt(np.sum([(x/8)**2 for x in sds])))
print(np.mean([(x/8)**2 for x in sds]))

df = pd.read_csv("data/image_scores.csv")
print(df.columns)
sds = []
for col_name in ['valsd', 'arosd', 'dom1sd']:
    sds = sds + [float(x) for x in df[col_name] if x !='.']

print(np.mean([(x/8)**2 for x in sds]))