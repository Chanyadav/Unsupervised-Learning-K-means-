import scipy.io
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the dataset AllSamples
data_set = scipy.io.loadmat("C:/Users/Chandan Yadav/Downloads/AllSamples.mat")

X = data_set["AllSamples"]

k_dist = [2, 3, 4, 5, 6, 7, 8, 9, 10]
s_lot = []
sum_dist = 0
# Convert the dataset to a Pandas DataFrame
x1 = pd.DataFrame(X)

# for loop to calculate the centroids for each value of k from k = 2-10
for k in range(2, 11):
    p = 2

    # Strategy 2: Pick the first centroid randomly
    centroids = {
        1: [x1[0][np.random.randint(0, 299)], x1[1][np.random.randint(0, 299)]]
    }

    # Strategy 2: for the i-th center (i>1), choose a sample (among all possible samples) such that the average
    # distance of this chosen one to all previous (i-1) centers is maximal
    for i in range(0, k - 1):
        averageDistanceFromAllCentroids = {}
        for value in X:
            distanceFromCentroids = [np.linalg.norm(value - [centroids.get(b)]) for b in centroids.keys()]
            averageDistanceFromAllCentroids[tuple(value)] = np.mean(distanceFromCentroids)
        centroids.update({p: list(max(averageDistanceFromAllCentroids, key=averageDistanceFromAllCentroids.get))})
        p = p + 1
    print("Initial centroids", centroids)


    # Assignment function used to assign each point to the nearest centroid
    def assignment(x1, centroids):
        for i in centroids.keys():
            x1['distance_from_{}'.format(i)] = (
                np.sqrt(
                    (x1[0] - centroids[i][0]) ** 2 +
                    (x1[1] - centroids[i][1]) ** 2
                )
            )
        # Calculate the distance of each point from every centroids and assign it to the closest one
        centroid_distance_closest = ['distance_from_{}'.format(i) for i in centroids.keys()]
        x1['closest'] = x1.loc[:, centroid_distance_closest].idxmin(axis=1)
        x1['closest'] = x1['closest'].map(lambda x: int(x.lstrip('distance _from_')))
        x1['color'] = x1['closest'].map(lambda x: 'b')
        return x1


    # First assignment step performed to assign the points to the calculated centroids

    x1 = assignment(x1, centroids)


    # Update function used to update the centroids. Calculate the mean of the each cluster and calculate the new centroids
    def update(k):
        for i in centroids.keys():
            if x1[x1['closest'] == i].size != 0:
                centroids[i][0] = np.mean(x1[x1['closest'] == i][0])
                centroids[i][1] = np.mean(x1[x1['closest'] == i][1])

        return k


    # First update step performed to update the first set of Centroids
    centroids = update(centroids)

    # while loop to iterate over the same above steps until the centroids are converge
    while True:
        closest_centroids = x1['closest'].copy(deep=True)
        centroids = update(centroids)
        x1 = assignment(x1, centroids)
        # break when the centroids don't change any more
        if closest_centroids.equals(x1['closest']):
            break
    print("Final Centroids", centroids)
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(x1[0], x1[1], color=x1['color'], alpha=0.5, edgecolor='g')

    for i in centroids.keys():
        plt.scatter(*centroids[i], color='red')

    plt.xlim(0, 10)
    plt.ylim(-1, 10)
    plt.xlabel('x---->')
    plt.ylabel('y---->')
    plt.title('Scatter plot')
    plt.show()

    # Calculation of the objective function

    sum_dist = 0
    for i in centroids.keys():
        sum_dist = sum_dist + np.sum((x1[x1['closest'] == i][0] - centroids[i][0]) ** 2) + np.sum(
            (x1[x1['closest'] == i][1] - centroids[i][1]) ** 2)

    s_lot.append(sum_dist)

# Plot the Objective function graph
print("The K values are", k_dist)
print("The variance for each k", s_lot)
plt.plot(k_dist, s_lot)
plt.scatter(k_dist, s_lot)
plt.title('Strategy 2 - Objective Function vs Number of clusters Graph')
plt.xlabel('number of clusters k')
plt.ylabel('Objective function value')







