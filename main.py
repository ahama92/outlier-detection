import pandas
import matplotlib.pyplot
import numpy
import seaborn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
    
def support_vector_machine_based(threshold=0.005):
    data = numpy.log10(df) # You can change this to data=df as well and see the results. 
    data=pandas.DataFrame(data)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pandas.DataFrame(np_scaled)
    # train oneclassSVM 
    model = OneClassSVM(nu=0.005, kernel="rbf", gamma=threshold)
    model.fit(data)
    df['support_vector_machine_based']=pandas.Series(model.predict(data))
    anomaly=df.loc[df['support_vector_machine_based']==-1 , ['residual']]
    anomaly=anomaly.loc[anomaly['residual']>df['residual'].mean()]
    
    return anomaly

def isolation_forest(threshold=0.002):
    data = numpy.log10(df['residual']) # You can change this to data=df as well and see the results. 
    data=pandas.DataFrame(data)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(data)
    data = pandas.DataFrame(np_scaled)
    # train isolation forest
    model =  IsolationForest(contamination=threshold)
    model.fit(data) 
    df['isolation_forest'] = pandas.Series(model.predict(data))
    anomaly=df.loc[df['isolation_forest']==-1 , ['residual']]
    anomaly=anomaly.loc[anomaly['residual']>df['residual'].mean()]
    
    return anomaly
    
def main():
    fig, axes = matplotlib.pyplot.subplots(nrows=2,figsize=(10,6))
    for ax, func in zip(axes, [support_vector_machine_based, isolation_forest]):
        anomaly = func()
        ax.plot(df.index,df['residual'], color='blue', label = 'Normal')
        ax.scatter(anomaly.index,anomaly['residual'], color='red', label = 'Anomaly')

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
    axes[0].set_title('Support Vector Machine-Based Outliers', **kwargs)
    axes[1].set_title('Isolation Forest Outliers', **kwargs)
    fig.suptitle('Iteration = {}'.format(i+1), size=14)

N=15
data=[]
for i in range(0,N):
    df=pandas.read_csv('./data/res'+str(i)+'.csv',skiprows=2,header=None)
    df.rename(columns={list(df)[0]:'residual'},inplace=True)
    df=df.abs()
    
    main()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


## Support Vector Machine-Based Anomaly Detection
# for i in range(0,N):
#     df=pandas.read_csv('./data/res'+str(i)+'.csv',skiprows=2,header=None)
#     df.rename(columns={list(df)[0]:'residual'},inplace=True)
#     df=df.abs()
#     data = numpy.log10(df) # You can change this to data=df as well and see the results. 
#     scaler = sklearn.preprocessing.StandardScaler()
#     np_scaled = scaler.fit_transform(data)
#     data = pandas.DataFrame(np_scaled)
#     # train oneclassSVM 
#     model = sklearn.svm.OneClassSVM(nu=0.005, kernel="rbf", gamma=0.005)
#     model.fit(data)
#     df['anomaly']=pandas.Series(model.predict(data))
#     anomaly=df.loc[df['anomaly']==-1 , ['residual']]
#     anomaly=anomaly.loc[anomaly['residual']>df['residual'].mean()]
#     # visualization
#     fig, ax = matplotlib.pyplot.subplots(figsize=(10,6))
#     ax.plot(df.index,df['residual'], color='blue', label = 'Normal')
#     ax.scatter(anomaly.index,anomaly['residual'], color='red', label = 'Anomaly')
#     ax.text(0,max(df['residual']),'Iteration '+str(i+1))
#     matplotlib.pyplot.legend()
#     matplotlib.pyplot.show();


## Isolation Forest for Anomaly Detection ====================================
# for i in range(0,N):
#     df=pandas.read_csv('./data/res'+str(i)+'.csv',skiprows=2,header=None)
#     df.rename(columns={list(df)[0]:'residual'},inplace=True)
#     df=df.abs()
#     data = numpy.log10(df)
#     scaler = sklearn.preprocessing.StandardScaler()
#     np_scaled = scaler.fit_transform(data)
#     data = pandas.DataFrame(np_scaled)
#     # train isolation forest
#     outliers_fraction=0.005
#     model =  sklearn.ensemble.IsolationForest(contamination=outliers_fraction)
#     model.fit(data) 
#     df['anomaly'] = pandas.Series(model.predict(data))
#     anomaly=df.loc[df['anomaly']==-1 , ['residual']]
#     anomaly=anomaly.loc[anomaly['residual']>df['residual'].mean()]
#     # visualization
#     fig, ax = matplotlib.pyplot.subplots(figsize=(10,6))
#     ax.plot(df.index,df['residual'], color='blue', label = 'Normal')
#     ax.scatter(anomaly.index,anomaly['residual'], color='red', label = 'Anomaly')
#     ax.text(0,max(df['residual']),'Iteration '+str(i))
#     matplotlib.pyplot.legend()
#     matplotlib.pyplot.show();


