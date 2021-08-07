import pandas
import matplotlib.pyplot
import sklearn.cluster
import mpl_toolkits.mplot3d.axes3d
import numpy
import sklearn.preprocessing
import sklearn.ensemble

N=15

data=[]
text=[]

for i in range(N):
    df=pandas.read_csv('./data/res'+str(i)+'.csv',skiprows=2,header=None)
    df.rename(columns={list(df)[0]:'residual'},inplace=True)
    df=df.abs()
    data = numpy.log10(df)
    scaler = sklearn.preprocessing.StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pandas.DataFrame(np_scaled)
    # train isolation forest
    outliers_fraction=0.005
    model =  sklearn.ensemble.IsolationForest(contamination=outliers_fraction)
    model.fit(data) 
    df['anomaly'] = pandas.Series(model.predict(data))
    anomaly=df.loc[df['anomaly']==-1 , ['residual']]
    anomaly=anomaly.loc[anomaly['residual']>df['residual'].mean()]
    # visualization
    fig, ax = matplotlib.pyplot.subplots(figsize=(10,6))
    ax.plot(df.index,df['residual'], color='blue', label = 'Normal')
    ax.scatter(anomaly.index,anomaly['residual'], color='red', label = 'Anomaly')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show();

# df=pandas.DataFrame(data,columns=text)
# df.info()
# print(data)
# print(columns)



# df=pandas.read_csv('./data/sol'+str(11)+'.csv',skiprows=2,header=None)
# df.abs().plot()
# matplotlib.pyplot.xlabel('Cell ID')
# matplotlib.pyplot.ylabel('Absolute Residual')

# data = df.abs()
# n_cluster = range(1, 20)
# kmeans = [sklearn.cluster.KMeans(n_clusters=i).fit(data) for i in n_cluster]
# scores = [kmeans[i].score(data) for i in range(len(kmeans))]
# fig, ax = matplotlib.pyplot.subplots(figsize=(10,6))
# ax.plot(n_cluster, scores)
# matplotlib.pyplot.xlabel('Number of Clusters')
# matplotlib.pyplot.ylabel('Score')
# matplotlib.pyplot.title('Elbow Curve')
# matplotlib.pyplot.show();

# X = df.abs()
# X = X.reset_index(drop=True)
# km = sklearn.cluster.KMeans(n_clusters=10)
# km.fit(X)
# km.predict(X)
# labels = km.labels_
# fig = matplotlib.pyplot.figure(1, figsize=(7,7))
# ax = mpl_toolkits.mplot3d.axes3d.Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
# ax.scatter(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2],c=labels.astype(numpy.float), edgecolor="k")
# ax.set_xlabel("price_usd")
# ax.set_ylabel("srch_booking_window")
# ax.set_zlabel("srch_saturday_night_bool")
# matplotlib.pyplot.title("K Means", fontsize=14);

# df=pandas.read_csv('./data/res'+str(1)+'.csv',skiprows=2,header=None)
# df.rename(columns={list(df)[0]:'residual'},inplace=True)
# df=df.abs()
# data = numpy.log10(df)
# scaler = sklearn.preprocessing.StandardScaler()
# np_scaled = scaler.fit_transform(data)
# data = pandas.DataFrame(np_scaled)
# # train isolation forest
# outliers_fraction=0.01
# model =  sklearn.ensemble.IsolationForest(contamination=outliers_fraction)
# model.fit(data) 
# df['anomaly'] = pandas.Series(model.predict(data))
# anomaly=df.loc[df['anomaly']==-1 , ['residual']]
# anomaly=anomaly.loc[anomaly['residual']>df['residual'].mean()]
# # visualization
# fig, ax = matplotlib.pyplot.subplots(figsize=(10,6))
# ax.plot(df.index,df['residual'], color='blue', label = 'Normal')
# ax.scatter(anomaly.index,anomaly['residual'], color='red', label = 'Anomaly')
# matplotlib.pyplot.legend()
# matplotlib.pyplot.show();






