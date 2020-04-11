import scipy.io
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the AllSamples dataset
data_set = scipy.io.loadmat("C:/Users/Chandan Yadav/Downloads/AllSamples.mat")

X = data_set["AllSamples"]

# List of number of values for k
k_dist = [2, 3, 4, 5, 6, 7, 8, 9, 10]
s_lot = []
sum_dist = 0;

# Convert the ALlSamples dataset into a pandas Dataframe
x1 = pd.DataFrame(X)
print(x1)
# Plot the scatter plot for the given distribution of data
plt.scatter(x1[0], x1[1], color='b', alpha=0.5, edgecolor='g')
plt.title("Scatter plot of the given distribution")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# loop to calculate the centroids for k values from 2-10
for k in range(2, 11):
    # Strategy 1: randomly pick the initial centers from the given samples.
    centroids = {
        i + 1: [x1[0][np.random.randint(0, 299)], x1[1][np.random.randint(0, 299)]]
        for i in range(k)
    }

    # Print the initial Centroids
    print("the initial centroids are")
    print(centroids)


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


    def update(k):
        for i in centroids.keys():
            centroids[i][0] = np.mean(x1[x1['closest'] == i][0])
            centroids[i][1] = np.mean(x1[x1['closest'] == i][1])
        return k


    # First update step performed to update the first set of Centroids
    centroids = update(centroids)

    # while loop to perform the above mentioned steps in a iterative fashion until the centroids converge.
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
plt.title('Strategy 1 - Objective Function vs Number of clusters Graph')
plt.xlabel('number of clusters k')
plt.ylabel('Objective function value')






