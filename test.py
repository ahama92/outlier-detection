import pandas

df=pandas.read_csv('./data/burgers-test-1/res'+str(0)+'.csv',skiprows=2,header=None)
df.rename(columns={list(df)[0]:'residual'},inplace=True)
# df=df.abs()

a=df>0