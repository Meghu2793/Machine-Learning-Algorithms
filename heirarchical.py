import numpy as np

def func(arr_):
    array2 = arr_    
    [min_x,min_y] = divmod(np.argmin(array2), arr_.shape[1])
    val_one = dendogram_chart[min_x]
    val_two = dendogram_chart[min_y]
    three = ([val_one,val_two])
    del(dendogram_chart[min_x])
    del(dendogram_chart[min_y-1])
    dendogram_chart.insert(min_x,three)
    newarray = np.delete(array2,min_y,axis=0)
    newarray = np.delete(newarray,min_y,axis=1)
    #print(newarray.shape)

    for i in range(0,len(newarray[:])):
        newarray[min_x,i] = np.min([arr_[i,min_x],arr_[i,min_y]])
        newarray[i,min_x] = np.min([arr_[min_x,i],arr_[min_y,i]])
        newarray[min_x,min_x] = 10000000
    
    if(newarray.shape[0] > 1):
        print(newarray)
        func(newarray)
    else:
        print(newarray)
        print(dendogram_chart)


def hierarchical_cluster(data):
    
    for i in range(0,len(data[:])):
        for j in range(0,len(data[:])):
            arr_[i][j]=np.linalg.norm(data[i,:] - data[j,:])
            if(i==j):
                arr_[i][j]=10000000
    return arr_
        
    
        
data = np.genfromtxt("SCLC_study_output_filtered_2.csv",delimiter=",")
data = data[1:,1:]
arr_ = np.zeros([data.shape[0],data.shape[0]])
dendogram_chart = range(data.shape[0])
print(dendogram_chart)
arr_=hierarchical_cluster(data)
func(arr_)