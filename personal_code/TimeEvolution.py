from Clustering import *

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
from Get_real_values import read_real_parameters_file


def unite_cluster(leaves, plan1, plan2):
    # Associate clusters so 1 cluster is equal to 1 leaf
    # The leaves object returned is reverse sorted by the z position
    # Each row has the following data : x, y, z, quarter, label, original order
    centerClusters = np.zeros((len(leaves), 6))
    centerClusters[:, 4] = -1
    centerClusters[:, 5] = np.arange(len(centerClusters))
    for i in range(len(leaves)):
        if len(leaves[i]) > 1:
            x, y, z = get_center_cluster(leaves[i])
            centerClusters[i, :5] = [x, y, z, -1, -1]

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
    # plot_data(centerClusters[:, 0:3], centerClusters[:, 4])
    maxInd = np.amax(centerClusters[:, 4])
    # centerClusters = centerClusters[centerClusters[:, 5].argsort()]  # Remove previous sorting so the order is the
    # same as leaves
    newLeaves = []
    for i in range(int(maxInd) + 1):  # +1 because the last index needs to be maxInd
        newLeaves.append([])

    for i in range(len(leaves)):
        if len(leaves[int(centerClusters[i, 5])]) != 1:
            if len(newLeaves[int(centerClusters[i, 4])]) == 0:
                newLeaves[int(centerClusters[i, 4])] = leaves[int(centerClusters[i, 5])]
            else:
                newLeaves[int(centerClusters[i, 4])] = np.append(newLeaves[int(centerClusters[i, 4])], leaves[int(centerClusters[i, 5])], axis=0)
    for i in range(len(newLeaves)):  # Supposed to be useless as newleaves has no empty list
        if(len(newLeaves[i])) == 0:
            newLeaves[i] = [1]
    quarterList = centerClusters[:, 3]
    quarterList2 = []
    for i in range(len(quarterList)):
        if quarterList[i] != 0:
            quarterList2.append(quarterList[i])
    quarterList3 = [quarterList2[0]]
    for i in range(1, len(quarterList2)):
        if quarterList2[i] != quarterList2[i-1]:
            quarterList3.append(centerClusters[i, 3])
    return newLeaves, quarterList3


def plot_leaf_number_evolution(plantAllTime, time):
    points = np.zeros((time - 1, 2))  # The data that will be plot
    points[:, 0] = np.arange(1, len(points) + 1)  # The x axis is just 1, 2, 3, ..., time
    for t in plantAllTime:
        nb_leaves = len(plantAllTime[t]) - 1  # The plant dictionary for a precise time is just : height, leaf 1, leaf 2, ...
        points[t - 1, 1] = nb_leaves
    plt.plot(points[:, 0], points[:, 1])
    plt.ylabel('Number of leaves')
    plt.xlabel('Time')
    plt.title('Evolution of leaf number for basil plant')
    # plt.show()


def plot_height_evolution(plantAllTime, time):
    points = np.zeros((time - 1, 2))  # The data that will be plot
    points[:, 0] = np.arange(1, len(points) + 1)  # The x axis is just 1, 2, 3, ..., time
    for t in plantAllTime:
        points[t - 1, 1] = plantAllTime[t]['Height']
    plt.plot(points[:, 0], points[:, 1])
    plt.ylabel('Height (cm)')
    plt.xlabel('Time')
    plt.title('Evolution of height for basil plant')
    # plt.show()


def plot_leaf_evolution(plantAllTime, time, characteristic, leafNumber, label):
    points = np.zeros((time - 1, 2))  # The data that will be plot
    points[:, 0] = np.arange(1, len(points) + 1)  # The x axis is just 1, 2, 3, ..., time
    for t in plantAllTime:
        try:
            leaf = plantAllTime[t][leafNumber]
        except:
            leaf = {
                "Leaf length": 0,
                "Width": 0,
                "Angle": 0,
                "Node height": 0,
                "Hidden": 0,
                "Main axis": 0,
                "Total length": 0,
                "Quarter": 0
            }
        points[t - 1, 1] = leaf[characteristic]
    plt.plot(points[:, 0], points[:, 1])
    # plt.ylabel(characteristic)
    # plt.xlabel('Time')
    # plt.title('Evolution of {} for leaf {} basil plant'.format(characteristic, leafNumber))


def evolution(time):
    # for i in range(44, 45):
    #     coordinates = read_data(i, "basil")
    #     dist = np.zeros(len(coordinates))
    #     for j in range(len(coordinates)):
    #         dist[j] = np.linalg.norm(coordinates[j, :2])
    #     # plot_data(coordinates[:-2], coordinates[:-2, 2])
    #     plot_data(coordinates[:-2], dist[:-2])
    #     leaves, nb_classes, pred = get_leaves(coordinates)
    #     mainAxis1 = [-0.46, -2.95, 0]  # To do : implement the main axes retrieving at the first time steps
    #     mainAxis2 = [2.95, -0.46, 0]
    #     leaves = separate_united_leaves(leaves)
    #     leaves, centerLeaves = unite_cluster(leaves, mainAxis1, mainAxis2)
    #     ncoordinates = get_numpy_array(leaves)
    #     plot_data(ncoordinates, ncoordinates[:, 3])

        # leaves = separate_united_leaves(leaves)
        # ncoordinates = get_numpy_array(leaves)
        # plot_data(ncoordinates, ncoordinates[:, 3])
        #
        # mainAxis1 = [-0.46, -2.95, 0]  # To do : implement the main axes retrieving at the first time steps
        # mainAxis2 = [2.95, -0.46, 0]
        # leaves, centerLeaves = unite_cluster(leaves, mainAxis1, mainAxis2)
        # ncoordinates = get_numpy_array(leaves)
        # plot_data(ncoordinates, ncoordinates[:, 3])
        # test = 3
        # plot_data(leaves[test], leaves[test][:, 2])
        # get_hidden_leaf_characteristics3(leaves[test])
    print('mark')
    # # for i in range(38, 41):
    # #     nc = read_data(i)
    # #     leaves, nb_clusters, predictions = get_leaves(nc)
    # #     if nb_clusters > 1:
    # #         un, deux = getUpperLeaves(leaves)
    # #         leaves = separate_united_leaves(leaves)
    # #         if len(leaves) != 0:
    # #             ncoord = get_numpy_array(leaves)
    # #             plot_data(ncoord, ncoord[:, 3])
    # #             unite_cluster(leaves, [-0.46, -2.95, 0], [2.95, -0.46, 0])
    # #             # ncoord2 = np.copy(leaves[0])
    # #             # ncoord2 = np.append(ncoord2, np.ones((len(leaves[0]), 1))*0, axis=1)
    # #             # for j in range(1, len(leaves)):
    # #             #     if len(leaves[j]) != 1:
    # #             #         temp = np.copy(leaves[j])
    # #             #         # Adding a column which is the label
    # #             #         temp = np.append(temp, np.ones((len(leaves[j]), 1))*j, axis=1)
    # #             #         ncoord2 = np.append(ncoord2, temp, axis=0)
    # #             # plot_data(ncoord2[:, :3], ncoord2[:, 3])
    # nc = read_data(40)
    # leaves, nb_clusters, predictions = get_leaves(nc)
    # # plot_data(leaves[0], leaves[0][:, 0])
    # get_parameters(leaves[0])
    # nc2 = get_numpy_array(leaves)
    # plot_data(nc2, nc2[:, 3])
    # leaves2 = separate_united_leaves(leaves)  ## Attention, erreur dans le reclustering
    # ncoord = get_numpy_array(leaves2)
    # # wtlist = []
    # # wt = []
    # # for i in range(len(leaves)):
    # #     wtlist.append(leaves[i] == leaves2[i])
    # #     wt.append(np.all(leaves[i] == leaves2[i]))
    # # wtflist = (nc2 == ncoord)
    # # wtf = np.all(nc2 == ncoord)
    # plot_data(ncoord, ncoord[:, 3])
    # newleaves = unite_cluster(leaves2, [-0.46, -2.95, 0], [2.95, -0.46, 0])
    # newcoord = get_numpy_array(newleaves)
    # plot_data(newcoord, newcoord[:, 3])
    # nb_leaf = 0
    # mainAxis1 = np.zeros(3)
    # mainAxis2 = np.zeros(3)
    plantAllTime = {}
    firstTry = True
    for dt in range(1, time):
        ncoord = read_data(dt, 'basil2')

        # Create the dictionary with the key parameter
        plantChar = {
            "Height": 0
        }

        totalHeight = get_height(ncoord)
        plantChar["Height"] = totalHeight

        # Change data format and using DBSCAN to get clusters
        leaves, nb_clusters, predictions = get_leaves(ncoord)

        if len(leaves) != 0:
            # if False:
            # # if nb_leaf == 0:
            #     if nb_clusters >= 2:
            #         nb_leaf = 2
            #         centers = list()
            #         for leaf in leaves:
            #             if len(leaf) > 1:
            #                 x, y, _ = get_center_cluster(leaf)
            #                 centers.append(np.abs(x) + np.abs(y))
            #         indexStem = centers.index(min(centers))
            #
            #         plans = []
            #         leafCounter = 1
            #         for i in range(len(leaves)):
            #             if i != indexStem:
            #                 length, width, angle, z, leafMainAxis, _ = get_parameters(leaves[i])
            #                 xmean, ymean, _ = get_center_cluster(leaves[i])
            #                 plantChar[str(leafCounter)] = {
            #                     "length": length,
            #                     "width": width,
            #                     "angle": angle,
            #                     "nodeHeight": z
            #                 }
            #
            #                 plan1 = np.array([xmean * math.cos(math.pi/4) - ymean * math.sin(math.pi/4),  # 45° rotation
            #                                   xmean * math.sin(math.pi / 4) + ymean * math.cos(math.pi / 4), 0])
            #                 plan2 = np.array([xmean * math.cos(-math.pi/4) - ymean * math.sin(-math.pi/4),  # -45° rotation
            #                                   xmean * math.sin(-math.pi / 4) + ymean * math.cos(-math.pi / 4), 0])
            #
            #                 plans.append(plan1)
            #                 plans.append(plan2)
            #                 leafCounter += 1
            #         mainAxis1 = (plans[0] + plans[2])/2
            #         mainAxis2 = (plans[1] + plans[3])/2
            #         break
            #     else:
            #         print('There is no leaves on time ' + str(dt) + ' !')

            # There is the beginning of the loop over time when the situation is stable
            # (i.e. there are at least 2 leaves on the basil)
            print('another mark')
            # leaves = separate_united_leaves(leaves)
            # c = get_numpy_array(leaves)
            # plot_data(c, c[:, 3])
            # mainAxis1 = [-0.46, -2.95, 0]  # To do : implement the main axes retrieving at the first time steps
            # mainAxis2 = [2.95, -0.46, 0]
            # leaves, centerLeaves = unite_cluster(leaves, mainAxis1, mainAxis2)
            print('pause')

            if dt >= 38:
                print('debug stop')

            mainAxis1 = [-0.46, -2.95, 0]  # To do : implement the main axes retrieving at the first time steps
            mainAxis2 = [2.95, -0.46, 0]
            mainAxis1 = [0.888, 2.6385, 0]  # To do : implement the main axes retrieving at the first time steps
            mainAxis2 = [2.6385, -0.888, 0]  # Main axis for basil 2
            # if dt >= 38:
            #     coord = get_numpy_array(leaves)
            #     plot_data(coord, coord[:, 3])
            leaves = separate_united_leaves(leaves)
            # if dt >= 38:
            #     coord = get_numpy_array(leaves)
            #     plot_data(coord, coord[:, 3])
            leaves, centerLeaves = unite_cluster(leaves, mainAxis1, mainAxis2)
            # if dt >= 38:
            #         coord = get_numpy_array(leaves)
            #         plot_data(coord, coord[:, 3])

            if firstTry:
                for i in range(len(leaves)):
                    leafLength, width, angle, zNode, leafMainAxis, length = get_parameters(leaves[i])
                    isHidden = False
                    leafDict = {
                        "Leaf length": leafLength,
                        "Width": width,
                        "Angle": angle,
                        "Node height": zNode,
                        "Hidden": isHidden,
                        "Main axis": leafMainAxis,
                        "Total length": length,
                        "Quarter": centerLeaves[i]
                    }
                    if i == 0:
                        for j in range(len(leaves)):
                            plantChar[j + 1] = {}
                            keyList = list(plantChar.keys())
                    plantChar[keyList[-1] - i] = leafDict
                    firstTry = False
            else:
                oldPlant = plantAllTime.get(dt-1)
                oldKeyList = list(oldPlant)
                newFloor = False
                for i in range(len(leaves)):
                    if len(leaves[i]) > 1:
                        if oldPlant != None:
                            oldKeyList = list(oldPlant.keys())
                        if i < 4:  # The leaf cannot be hidden
                            leafLength, width, angle, zNode, leafMainAxis, length = get_parameters(leaves[i])
                            isHidden = False
                        else:
                            # Be aware that the length here is with the petiole
                            width, angle, zNode, leafMainAxis, length = get_hidden_leaf_characteristics3(leaves[i])
                            leafLength = 0
                            isHidden = True

                        try:
                            quarter = centerLeaves[i]
                        except:
                            quarter = 0
                        leafDict = {
                            "Leaf length": leafLength,
                            "Width": width,
                            "Angle": angle,
                            "Node height": zNode,
                            "Hidden": isHidden,
                            "Main axis": leafMainAxis,
                            "Total length": length,
                            "Quarter": quarter
                        }
                        if i < 1:  # The first leaf in the list (the upper one)
                            if (oldPlant[oldKeyList[-1]]['Quarter'] - centerLeaves[i]) % 2 == 1:
                                for j in range(oldKeyList[-1] + 2):  # There are 2 more leaves than before
                                    plantChar[j + 1] = {}
                                actualKeyList = list(plantChar.keys())
                                newFloor = True
                            else:  # The number of leaves remains the same as before
                                for j in range(oldKeyList[-1]):
                                    plantChar[j + 1] = {}
                                actualKeyList = list(plantChar.keys())

                        gap = 0
                        if newFloor == True:
                            gap = 2  # If there is a new floor, the end of the old list is 2 units lesser than the new one
                        if i < 2 and newFloor == True:
                            plantChar[actualKeyList[-1] - i] = leafDict
                        else:
                            # The following if else are there to make equal the leaves between two step times
                            error1 = leafDict['Quarter']
                            error2 = oldPlant[oldKeyList[-1] - i + gap]['Quarter']
                            if leafDict['Quarter'] == oldPlant[oldKeyList[-1] - i + gap]['Quarter']:
                                plantChar[actualKeyList[-1] - i] = leafDict
                                if isHidden == True:
                                    plantChar[actualKeyList[-1] - i]['Leaf length'] = oldPlant[oldKeyList[-1] - i + gap]['Leaf length']
                            else:
                                # We need to avoid storing a leaf on the wrong floor
                                if i % 2 == 0:  # There are 2 leaves per floor, the first treated is when i is even and the second treated is when i is odd
                                    plantChar[actualKeyList[-1] - i - 1] = leafDict  # The corresponding leaf between t and t-1 are not on the same position
                                    if isHidden == True:
                                        plantChar[actualKeyList[-1] - i - 1]['Leaf length'] = \
                                        plantAllTime[dt - 1][actualKeyList[-1] - i - 1]['Leaf length']
                                        if width < plantAllTime[dt-1][actualKeyList[-1] - i - 1]['Width']:
                                            plantChar[actualKeyList[-1] - i - 1]['Width'] = \
                                                plantAllTime[dt-1][actualKeyList[-1] - i - 1]['Width']
                                        if length < plantAllTime[dt-1][actualKeyList[-1] - i - 1]['Total length']:
                                            plantChar[actualKeyList[-1] - i - 1]['Total length'] = \
                                                plantAllTime[dt - 1][actualKeyList[-1] - i - 1]['Total length']
                                else:  # The leaf is the second one of the floor
                                    plantChar[actualKeyList[-1] - i + 1] = leafDict  # The corresponding leaf between t and t-1 are not on the same position
                                    if isHidden == True:
                                        plantChar[actualKeyList[-1] - i + 1]['Leaf length'] = \
                                        plantAllTime[dt - 1][actualKeyList[-1] - i + 1]['Leaf length']
                                        if width < plantAllTime[dt-1][actualKeyList[-1] - i + 1]['Width']:
                                            plantChar[actualKeyList[-1] - i + 1]['Width'] = \
                                                plantAllTime[dt-1][actualKeyList[-1] - i + 1]['Width']
                                        if length < plantAllTime[dt-1][actualKeyList[-1] - i + 1]['Total length']:
                                            plantChar[actualKeyList[-1] - i + 1]['Total length'] = \
                                                plantAllTime[dt - 1][actualKeyList[-1] - i + 1]['Total length']

            plantAllTime[dt] = plantChar
            print('Halleluja')
    plot_leaf_number_evolution(plantAllTime, time)
    # plot_height_evolution(plantAllTime, time)
    characteristic = 'Node height'
    leafNumber = 5
    # plot_leaf_evolution(plantAllTime, time, characteristic, leafNumber, 'estimation')
    real = read_real_parameters_file('Real_parameters_of_basil2.txt')
    real2 = {}
    for param in real:
        if type(param) == str:
            real2[param] = real[param]
        if type(param) != str:
            if param < 39:
                real2[param] = real[param]

    # plot_leaf_evolution(real2, time, characteristic, leafNumber, 'real')
    # plot_height_evolution(real2, time)
    plot_leaf_number_evolution(real2, time)
    # plt.ylabel(characteristic)
    plt.xlabel('Time')
    # plt.title('Evolution of {} for leaf {} basil plant'.format(characteristic, leafNumber))
    plt.show()




    print('First leave on t = ' + str(dt))

evolution(39)