import pandas
import matplotlib.pyplot
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
    
def density_based_spatial_clustering(eps=0.25,min_samples=10):
    outlier_detection = DBSCAN(eps=eps,min_samples=min_samples)
    df['density_based_spatial_clustering']=outlier_detection.fit_predict(data)
    anomaly=df.loc[df['density_based_spatial_clustering']==-1 , ['residual']]
    anomaly=anomaly.loc[anomaly['residual']>df['residual'].mean()]
    
    return anomaly
    
def support_vector_machine_based(threshold=0.005):
    model = OneClassSVM(nu=0.005, kernel="rbf", gamma=threshold)
    model.fit(data)
    df['support_vector_machine_based']=pandas.Series(model.predict(data))
    anomaly=df.loc[df['support_vector_machine_based']==-1 , ['residual']]
    anomaly=anomaly.loc[anomaly['residual']>df['residual'].mean()]
    
    return anomaly

def isolation_forest(threshold=0.002):
    model =  IsolationForest(contamination=threshold)
    model.fit(data) 
    df['isolation_forest'] = pandas.Series(model.predict(data))
    anomaly=df.loc[df['isolation_forest']==-1 , ['residual']]
    anomaly=anomaly.loc[anomaly['residual']>df['residual'].mean()]
    
    return anomaly
    
def main():
    func_list=[support_vector_machine_based,isolation_forest,density_based_spatial_clustering]
    fig, axes = matplotlib.pyplot.subplots(nrows=func_list.__len__(),figsize=(10,6))
    for ax, func in zip(axes,func_list):
        anomaly = func()
        ax.plot(df.index,df['residual'], color='blue', label = 'Normal')
        ax.scatter(anomaly.index,anomaly['residual'], color='red', label = 'Anomaly')

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
    for j in range(len(func_list)):
        axes[j].set_title(func_list[j].__name__, **kwargs)
        fig.suptitle('Iteration = {}'.format(i+1), size=14)

data=[]
for i in range(1,10):
    df=pandas.read_csv('./data/burgers-test-1/res'+str(i)+'.csv',skiprows=2,header=None)
    df.rename(columns={list(df)[0]:'residual'},inplace=True)
    df=df.abs()
    df[df<1e-20]=1e-20
    data = numpy.log10(df) # You can change this to data=df as well and see the results. 
    data=pandas.DataFrame(data)
    ## You can scale the data with the following three lines.
    # scaler = StandardScaler()
    # np_scaled = scaler.fit_transform(data)
    # data = pandas.DataFrame(np_scaled)
    
    main()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    