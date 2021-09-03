import sys
import pandas
import matplotlib.pyplot
from matplotlib import animation
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
    
def support_vector_machine_based(threshold=0.005):
    model = OneClassSVM(nu=0.005, kernel="rbf", gamma=threshold)
    model.fit(data)
    df['support_vector_machine_based']=pandas.Series(model.predict(data))
    anomaly=df.loc[df['support_vector_machine_based']==-1 , [1]]
    anomaly=anomaly.loc[anomaly[1]>df[1].mean()]
    
    return anomaly

def isolation_forest(threshold=0.001):
    model =  IsolationForest(contamination=threshold)
    model.fit(data) 
    df['isolation_forest'] = pandas.Series(model.predict(data))
    anomaly=df.loc[df['isolation_forest']==-1 , [1]]
    anomaly=anomaly.loc[anomaly[1]>df[1].mean()]
    
    return anomaly
    
def density_based_spatial_clustering(eps=0.25, min_samples=10):
    outlier_detection = DBSCAN(eps=eps,min_samples=min_samples)
    df['density_based_spatial_clustering']=outlier_detection.fit_predict(data)
    anomaly=df.loc[df['density_based_spatial_clustering']==-1 , [1]]
    anomaly=anomaly.loc[anomaly[1]>df[1].mean()]
    
    return anomaly

def solve(i=0):
    global df
    global data
    df=pandas.read_csv('./data/burgers-test-1/Burgers-solutionModes-'+str(i)+'.dat',sep=' ',skiprows=2,header=None)
    # print(df.size)
    df.rename(columns={list(df)[0]:'iteration'},inplace=True)
    df=df.dropna(axis=1)
    df=df.abs()
    # df[df<1e-20]=1e-20
    # data = numpy.log10(df[1]) # You can change this to data=df as well and see the results. 
    data=pandas.DataFrame(df[1])
    
    # return df, data

def update(iteration=0):
    solve(iteration)
    # for line_temp, scatter_temp, ax, func in zip(line, scatter, axes, func_list):
    for i in range(3):
        anomaly = func_list[i]()
        line[i*2].set_data(df['iteration'],df[1])
        line[i*2+1].set_data(anomaly.index,anomaly[1])
        axes[i].set_ylim([0,1.1*max(df[1])])
    
    return line

solve(0)
func_list=[support_vector_machine_based,isolation_forest,density_based_spatial_clustering]
fig, axes = matplotlib.pyplot.subplots(nrows=func_list.__len__(),figsize=(10,6))
line=[]
scatter=[]
for ax, func in zip(axes, func_list):
    anomaly = func()
    line_temp,=ax.plot(df['iteration'],df[1], color='blue', label = 'Normal')
    scatter_temp,=ax.plot(anomaly.index, anomaly[1], 'o', color='red', label = 'Anomaly')
    # print(scatter_temp.get_paths())
    ax.set_ylim([0,1.1*max(df[1])])
    line.append(line_temp)
    line.append(scatter_temp)
    
title=fig.suptitle('Iteration = {}'.format(1), size=14)
kwargs = dict(y=0.90, x=0.15, ha='left', va='top')
for j in range(len(func_list)):
    axes[j].set_title(func_list[j].__name__, **kwargs)

matplotlib.pyplot.legend()
matplotlib.pyplot.show()

anim = animation.FuncAnimation(fig, update, frames=50, interval=20, blit=True)
anim.save('test_animation.mp4', writer='ffmpeg', fps=2, extra_args=['-vcodec', 'libx264'])
    
    

   

# data=[]
# for i in range(0,20):
#     df=pandas.read_csv('./data/burgers-test-1/Burgers-solutionModes-'+str(i)+'.dat',sep=' ',skiprows=2,header=None)
#     # print(df.size)
#     df.rename(columns={list(df)[0]:'iteration'},inplace=True)
#     df=df.dropna(axis=1)
#     df=df.abs()
#     # df[df<1e-20]=1e-20
#     # data = numpy.log10(df[1]) # You can change this to data=df as well and see the results. 
#     data=pandas.DataFrame(df[1])
#     # data=pandas.DataFrame(data)
#     ## You can scale the data with the following three lines.
#     # scaler = StandardScaler()
#     # np_scaled = scaler.fit_transform(data)
#     # data = pandas.DataFrame(np_scaled)
    
#     # model = OneClassSVM(nu=0.005, kernel="rbf", gamma=0.005)
#     # model.fit(data)
#     # df['support_vector_machine_based']=pandas.Series(model.predict(data))
#     # anomaly=df.loc[df['support_vector_machine_based']==-1 , [1]]
#     # anomaly=anomaly.loc[anomaly[1]>df[1].mean()]
    
#     # matplotlib.pyplot.plot(df['iteration'],df[1])
#     # matplotlib.pyplot.scatter(anomaly.index,anomaly[1], color='red', label = 'Anomaly')
#     # sys.exit()
    
#     main()
    