import pandas as pd
df = pd.read_csv("guardian_corpus.csv")
sample = df.sample(n=500, random_state=42)
sample.to_csv("guardian_sample_500.csv", index=False)
