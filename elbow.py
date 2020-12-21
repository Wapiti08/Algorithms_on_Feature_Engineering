from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# calculate the WSS (Within-Cluster-Sum of Squared) an return the list
def WSS_cal(points, max_k):
    '''
        points: the x 
        max_k: the max k value for test
    '''
    distance_list = []
    # calculate the values for different k
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k).fit(points)
        # get the vector with shape [n_clusters, n_features]
        # features here are 2
        cen_features = kmeans.cluster_centers_
        # predict the closet clusters
        pre_clusters = kmeans.predict(points)

        # create the distance values to compare 
        cur_dis = 0
        # calculate "square" of Euclidean distance for each point
        for i in range(points.shape[0]):
            cur_center = cen_features[pre_clusters[i]]
            # ^x **2 + ^y **2
            cur_dis += (points[i,0] - cur_center[0])**2 + (points[i,1] - cur_center[1])**2
        
        distance_list.append(cur_dis)
    
    return distance_list

if __name__ == '__main__':

    # Create dataset with 5 random cluster centers and 1000 datapoints
    x, y = make_blobs(n_samples = 1000, centers = 5, n_features=2, shuffle=True, random_state=43)
    plt.scatter(x[:, 0], x[:, 1], s=50)
    # the shape of x will be (1000,2) --- (x_column, y_column)
    distance_list = WSS_cal(x, 7)
    k_list = range(1, 8)
    plt.figure()
    plt.plot(k_list, distance_list)
    plt.show()
