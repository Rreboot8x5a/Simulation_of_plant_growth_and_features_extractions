from Clustering import *

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import math


def unite_cluster(leaves, plan1, plan2):
    # Associate clusters so 1 cluster is equal to 1 leaf
    # The leaves object returned is reverse sorted by the z position
    # Each row has the following data : x, y, z, quarter, label, original order
    centerClusters = np.zeros((len(leaves), 5))
    centerClusters[:, 4] = -1
    for i in range(len(leaves)):
        if len(leaves[i]) > 1:
            x, y, z = getCenterCluster(leaves[i])
            centerClusters[i, :] = [x, y, z, -1, -1]

            # Define the quarter in which is the cluster
            bool1 = (plan1[0] * x + plan1[1] * y <= 0)
            bool2 = (plan2[0] * x + plan2[1] * y <= 0)
            # "Bool1 is True" does not work (probably for shady reasons with numpy)
            if (bool1 == True) and (bool2 == True):
                centerClusters[i, 3] = 1
            if (bool1 == True) and (bool2 == False):
                centerClusters[i, 3] = 2
            if (bool1 == False) and (bool2 == False):
                centerClusters[i, 3] = 3
            if (bool1 == False) and (bool2 == True):
                centerClusters[i, 3] = 4

    centerClusters = centerClusters[(-centerClusters[:, 2]).argsort()]  # Reverse sort by z coordinate
    actualQuarter = centerClusters[0, 3]
    label = 0
    # The clusters are sorted by z position, now we verify in which quarter belongs each center cluster
    # Then with this 2 pieces of information, we retrieve if 2 clusters are from the same leaf.
    for i in range(len(centerClusters)):
        if centerClusters[i, 3] != 0:
            # The quarter of the cluster is the same or the opposite as the actual one
            if (actualQuarter - centerClusters[i, 3]) % 2 == 0:
                if centerClusters[i, 3] == actualQuarter:
                    centerClusters[i, 4] = label
                else:
                    centerClusters[i, 4] = label + 1
            else:
                label += 2
                actualQuarter += 1  # The actual quarter is the one perpendicular to the previous one
                if actualQuarter == 5:
                    actualQuarter = 1  # Make a loop to not be out of bound

                # Do the same shit as before
                if (actualQuarter - centerClusters[i, 3]) % 2 == 0:
                    if centerClusters[i, 3] == actualQuarter:
                        centerClusters[i, 4] = label
                    else:
                        centerClusters[i, 4] = label + 1
    plot_data(centerClusters[:, 0:3], centerClusters[:, 4])
    maxInd = np.amax(centerClusters[:, 4])
    # centerClusters = centerClusters[centerClusters[:, 5].argsort()]  # Remove previous sorting
    newLeaves = []
    for i in range(int(maxInd) + 1):  # +1 because the last index needs to be maxInd
        newLeaves.append([])
    for i in range(len(leaves)):
        if len(leaves[i]) != 1:
            if len(newLeaves[int(centerClusters[i, 4])]) == 0:
                newLeaves[int(centerClusters[i, 4])] = leaves[i]
            else:
                newLeaves[int(centerClusters[i, 4])] = np.append(newLeaves[int(centerClusters[i, 4])], leaves[i], axis=0)
    for i in range(len(newLeaves)):
        if(len(newLeaves[i])) == 0:
            newLeaves[i] = [1]
    newcoord = get_numpy_array(newLeaves)
    plot_data(newcoord, newcoord[:, 3])
    return newLeaves



def evolution(time):
    # for i in range(38, 41):
    #     nc = read_data(i)
    #     leaves, nb_clusters, predictions = get_leaves(nc)
    #     if nb_clusters > 1:
    #         un, deux = getUpperLeaves(leaves)
    #         leaves = separate_united_leaves(leaves)
    #         if len(leaves) != 0:
    #             ncoord = get_numpy_array(leaves)
    #             plot_data(ncoord, ncoord[:, 3])
    #             unite_cluster(leaves, [-0.46, -2.95, 0], [2.95, -0.46, 0])
    #             # ncoord2 = np.copy(leaves[0])
    #             # ncoord2 = np.append(ncoord2, np.ones((len(leaves[0]), 1))*0, axis=1)
    #             # for j in range(1, len(leaves)):
    #             #     if len(leaves[j]) != 1:
    #             #         temp = np.copy(leaves[j])
    #             #         # Adding a column which is the label
    #             #         temp = np.append(temp, np.ones((len(leaves[j]), 1))*j, axis=1)
    #             #         ncoord2 = np.append(ncoord2, temp, axis=0)
    #             # plot_data(ncoord2[:, :3], ncoord2[:, 3])
    nc = read_data(40)
    leaves, nb_clusters, predictions = get_leaves(nc)
    # plot_data(leaves[0], leaves[0][:, 0])
    get_parameters(leaves[0])
    nc2 = get_numpy_array(leaves)
    plot_data(nc2, nc2[:, 3])
    leaves2 = separate_united_leaves(leaves)  ## Attention, erreur dans le reclustering
    ncoord = get_numpy_array(leaves2)
    plot_data(ncoord, ncoord[:, 3])
    unite_cluster(leaves, [-0.46, -2.95, 0], [2.95, -0.46, 0])
    nb_leaf = 0
    mainAxis1 = np.zeros(3)
    mainAxis2 = np.zeros(3)
    plantAllTime = {}
    for dt in range(1, time):
        ncoord = read_data(dt)

        # Create the dictionary with the key parameter
        plantChar = {
            "Height": 0
        }

        totalHeight = get_height(ncoord)
        plantChar["Height"] = totalHeight

        # Change data format and using DBSCAN to get clusters
        leaves, nb_clusters, predictions = get_leaves(ncoord)

        if len(leaves) != 0:
            if nb_leaf == 0:
                if nb_clusters >= 2:
                    nb_leaf = 2
                    centers = list()
                    for leaf in leaves:
                        if len(leaf) > 1:
                            x, y, _ = getCenterCluster(leaf)
                            centers.append(np.abs(x) + np.abs(y))
                    indexStem = centers.index(min(centers))

                    plans = []
                    leafCounter = 1
                    for i in range(len(leaves)):
                        if i != indexStem:
                            length, width, angle, z = get_parameters(leaves[i])
                            xmean, ymean, _ = getCenterCluster(leaves[i])
                            plantChar[leafCounter] = {
                                "length": length,
                                "width": width,
                                "angle": angle,
                                "nodeHeight": z
                            }

                            plan1 = np.array([xmean * math.cos(math.pi/4) - ymean * math.sin(math.pi/4),  # 45° rotation
                                              xmean * math.sin(math.pi / 4) + ymean * math.cos(math.pi / 4), 0])
                            plan2 = np.array([xmean * math.cos(-math.pi/4) - ymean * math.sin(-math.pi/4),  # -45° rotation
                                              xmean * math.sin(-math.pi / 4) + ymean * math.cos(-math.pi / 4), 0])

                            plans.append(plan1)
                            plans.append(plan2)
                            leafCounter += 1
                    mainAxis1 = (plans[0] + plans[2])/2
                    mainAxis2 = (plans[1] + plans[3])/2
                    break
                else:
                    print('There is no leaves on time ' + str(dt) + ' !')

            # There is the beginning of the loop over time when the situation is stable
            # (i.e. there are at least 2 leaves on the basil)
            else:
                leaves = separate_united_leaves(leaves)
                leaves = unite_cluster(leaves, mainAxis1, mainAxis1)
                for i in range(len(leaves)):
                    if i < 4:  # The leaf cannot be hidden
                        length, width, angle, zNode = get_parameters(leaves[i])
                        isHidden = False
                    else:
                        # Be aware that the length here is with the petiole
                        length, width, angle, zNode = get_hidden_leaf_characteristics(leaves[i])
                        isHidden = True
                    leafDict = {
                        "Length": length,
                        "Width": width,
                        "Angle": angle,
                        "Node height": zNode,
                        "Hidden": isHidden
                    }
                    plantChar[str(len(leaves) - i)] = leafDict


    print('First leave on t = ' + str(dt))

evolution(51)