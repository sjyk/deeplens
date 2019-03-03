import numpy as np
import pandas as pd

s = pd.Series([1,2,3,np.nan,6,8])
print("s: ", s)

dates = pd.date_range("20190201", periods=28, freq="1D")
print("dates: ", dates)

df = pd.DataFrame(np.random.randn(28, 4), index=dates, columns=list("ABCD"))
print("df: \n", df)
