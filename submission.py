from pathlib import Path
from submission import *

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

df_test = pd.read_csv(Path("../data") / "SampleSubmissionStage2.csv")
df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))



print(team_a_stats.head())
print(len(team_a_stats))
print(team_a_stats.columns)



print(match_features.shape)
print(match_features.columns)
