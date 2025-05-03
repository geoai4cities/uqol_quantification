import os
import pandas as pd
import numpy as np
from natsort import natsorted

filenames = natsorted(os.listdir('./datasets/Bhopal_RGB/images'))
df = pd.DataFrame(filenames)
df.to_csv('./datasets/Bhopal_RGB/all.txt', header=None, index=False)
# print(df.head())