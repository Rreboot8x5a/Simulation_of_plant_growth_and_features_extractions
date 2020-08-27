from Clustering import *

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import math


def evolution(time):
    nb_leaf = 0
    mainAxis1 = np.zeros(3)
    mainAxis2 = np.zeros(3)
    plantAllTime = {}
    for dt in range(1, time):
        ncoord = readData(dt)

        # Create the dictionary with the key parameter
        plantChar = {
            "Height": 0
        }

        # Change data format and using DBSCAN to get clusters
        leaves, nb_clusters, predictions = getLeaves(ncoord)

        totalHeight = getHeight(ncoord)
        plantChar["Height"] = totalHeight

        if nb_leaf == 0:
            if nb_clusters >= 2:
                nb_leaf = 2
                centers = list()
                for leaf in leaves:
                    x, y, _ = getCenterCluster(leaf)
                    centers.append(np.abs(x) + np.abs(y))
                indexStem = centers.index(min(centers))

                plans = []
                leafCounter = 1
                for i in range(len(leaves)):
                    if i != indexStem:
                        length, width, leafPCA, angle, z = getLengthWidth(leaves[i])
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
    print('First leave on t = ' + str(dt))

evolution(51)