import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import math


def split(arr, cond):
  return arr[cond], arr[~cond]


def get_leaves(ncoord):
    length, width, height, trueLength = 1280, 720, 50, 50
    dp = trueLength/length  # the length of a pixel if it hits the ground

    model = DBSCAN(eps=4 * dp, min_samples=15, n_jobs=4)
    model.fit_predict(ncoord)
    pred = model.fit_predict(ncoord)

    nb_classes = np.max(pred) + 1  # The label are going from 0 to n-1, so n is the number of label and is the max
    # value possible for a label
    leaves = [[]] * nb_classes
    if nb_classes > 1:  # If there is 0 class or 1 class, do nothing (1 class is just the top of the stem
        for i in range(len(ncoord)):
            if pred[i] != -1:
                if len(leaves[pred[i]]) == 0:
                    leaves[pred[i]] = [ncoord[i, :]]
                else:
                    leaves[pred[i]].append(ncoord[i, :])

        for i in range(nb_classes):
            leaves[i] = np.array(leaves[i])

        for i in range(len(leaves)):  # delete cluster without enough point
            # if len(leaves[i]) < 0.01 * len(ncoord):
            if len(leaves[i]) < 100:  # Removing the cluster with the stem tip
                leaves[i] = np.array([1])  # Just a definite array to check whether we keep the cluster or not

        nb_points_clean = 0
        for cluster in leaves:
            if len(leaves[i]) != 1:
                nb_points_clean += len(cluster)
        if nb_points_clean != 0:
            # ncoord2 = np.zeros((nb_points_clean, 3))
            # ncoord2 = np.zeros((nb_points_clean, 4))
            # indice = 0
            # for i in range(len(leaves)):
            #     ncoord2[indice:indice + len(leaves[i]), :-1] = leaves[i]
            #     ncoord2[indice:indice + len(leaves[i]), -1] = np.full((len(leaves[i])), i)
            #     indice += len(leaves[i])
            return leaves, nb_classes, pred
        else:  # There is at least one cluster, but those clusters are not big enough
            return [], nb_classes, []
    else:
        return [], nb_classes, []


def get_height(ncoord):
    zcoord = np.copy(ncoord[:, 2])
    zcoord.sort()
    height = zcoord[-2]
    return height


# def getUpperLeaves(leaves2):  # Can be more precise if we take the height of the beginning of the petiole
#     # (but only working if the main stem is totally vertical, or the point will not be guessed correctly)
#     zMeans = []
#     for i in range(len(leaves2)):
#         if len(leaves2[i]) != 1:
#             zMeans.append([i, np.sum(leaves2[i][:, 2]) / len(leaves2[i])])
#     if len(zMeans) != 0:
#         zMeans = np.array(zMeans)
#         zMeans = zMeans[zMeans[:, 1].argsort()]
#         return zMeans[-1, 0], zMeans[-2, 0]
#     else:
#         return -1, -1


def get_petiole_beginning(leaf):  # The leaf cloud must not be transformed by the PCA
    tempLeaf = np.copy(leaf)  # To avoid transformation on original leaf
    distFromCenter = np.sqrt(tempLeaf[:, 0]**2 + tempLeaf[:, 1]**2)
    leaf2 = np.append(np.arange(len(tempLeaf))[np.newaxis].T, tempLeaf, axis=1)
    leaf2 = np.append(leaf2, distFromCenter[np.newaxis].T, axis=1)
    leaf2 = leaf2[leaf2[:, 4].argsort()]  # Sort by distance from z axis
    indData = leaf2[0, :]
    index = leaf2[0, 0]
    z = leaf2[0, 3]
    return int(index), z


def get_parameters(leaf):
    index, zNode = get_petiole_beginning(leaf)
    pca = PCA(n_components=3)
    leafPCA = pca.fit_transform(X=leaf)
    # plot_data(leafPCA, leafPCA[:, 0])
    eigenVectorX2 = pca.components_[0]
    horiz = np.sqrt(eigenVectorX2[0] ** 2 + eigenVectorX2[1] ** 2)
    angle = np.abs(math.atan(eigenVectorX2[2]/horiz))
    angle2 = 90 - angle/math.pi * 180
    xLeaf = np.copy(leafPCA)  # A copy of the leaf after the PCA transform, will be sorted by the first axis value (x)
    yLeaf = np.copy(leafPCA)  # A copy of the leaf after the PCA transform, will be sorted by the second axis value (y)
    xLeaf = np.append(np.arange(len(xLeaf))[np.newaxis].T, xLeaf, axis=1)
    xLeaf = xLeaf[xLeaf[:, 1].argsort()]
    length = xLeaf[-1, 1] - xLeaf[0, 1]
    yLeaf = np.append(np.arange(len(yLeaf))[np.newaxis].T, yLeaf, axis=1)
    yLeaf = yLeaf[yLeaf[:, 2].argsort()]
    width = yLeaf[-1, 2] - yLeaf[0, 2]

    xPos, xPetiole, xLeaf2 = separate_petiole_from_leaf(xLeaf, index)  # xLeaf2 is a copy of xLeaf without the petiole
    xLeaf2 = xLeaf2[xLeaf2[:, 1].argsort()]
    leafLength = xLeaf[-1, 1] - xLeaf[0, 1]
    petioleLength = length - leafLength

    # subfig = go.Scatter3d(
    #     x=xLeaf[:, 1],
    #     y=xLeaf[:, 2],
    #     z=xLeaf[:, 3],
    #     mode='markers',
    #     marker=dict(
    #         size=2,
    #         # color=pred,
    #         color=xLeaf[:, 0],
    #         colorscale='Viridis',
    #         opacity=0.8
    #     )
    # )
    # fig = make_subplots(
    #     rows=1, cols=1,
    #     specs=[[{'type': 'surface'}]])
    # fig.add_trace(subfig)
    # fig.update_layout(scene_aspectmode='data', )
    # fig.show()

    return leafLength, width, angle2, zNode


def separate_petiole_from_leaf(xLeaf, beginIndex):
    print('CQC over 9000')
    # beginIndex = get_petiole_beginning(xLeaf)
    end = xLeaf[-1, 1]
    begin = xLeaf[0, 1]
    direction = 0
    beginningPoints = np.where(xLeaf[:, 1] <= begin + 0.3, xLeaf[:, 0], 0)
    endingPoints = np.where(xLeaf[:, 1] >= end - 0.3, xLeaf[:, 0], 0)
    if beginIndex in beginningPoints:  # the leaf is growing left to right direction
        direction = 1
    if beginIndex in endingPoints:  # the leaf is growing right to left direction,
        direction = -1  # so we need to swap ending and beginning
        temp = end
        end = begin
        begin = temp
    position = begin

    while position * direction <= end * direction:
        position += 0.1 * direction  # Adding 1 mm par step
        validPoints = np.where(xLeaf[:, 1] * direction <= position * direction, xLeaf[:, 2], 0)  # The Y position of points between xMin and position
        yMax = np.amax(validPoints)
        yMin = np.amin(validPoints)

        if yMax - yMin >= 0.3:
            break

    xPetiole, xLeaf = split(xLeaf, xLeaf[:, 1] * direction <= position * direction)
    # Due to the rectangle leaf, the petiole is not a line but a T
    # (because "position" has a max error of 1mm so we can cute the leaf to late)

    # subfig = go.Scatter3d(
    #     x=xLeaf[:, 1],
    #     y=xLeaf[:, 2],
    #     z=xLeaf[:, 3],
    #     mode='markers',
    #     marker=dict(
    #         size=2,
    #         # color=pred,
    #         color=xLeaf[:, 0],
    #         # color=xPetiole[:, 0],
    #         colorscale='Viridis',
    #         opacity=0.8
    #     )
    # )
    # fig = make_subplots(
    #     rows=1, cols=1,
    #     specs=[[{'type': 'surface'}]])
    # fig.add_trace(subfig)
    # fig.update_layout(scene_aspectmode='data', )
    # fig.show()
    return position, xPetiole, xLeaf


def getCenterCluster(cluster):  # Need cluster before any transformation (cluster coming from DBSCAN)
    x = np.sum(cluster[:, 0]) / len(cluster[:, 0])
    y = np.sum(cluster[:, 1]) / len(cluster[:, 1])
    z = np.sum(cluster[:, 2]) / len(cluster[:, 2])
    return x, y, z


# def uniteClusters(clusters, planNormal, nb_classes):
#     centers = np.zeros((nb_classes, 3))
#     for i in range(len(clusters)):
#         x, y, z = getCenterCluster(clusters[i])
#         centers[i, :] = [x, y, z]
#
#     # subfig = go.Scatter3d(
#     #     x=centers[:, 0],
#     #     y=centers[:, 1],
#     #     z=centers[:, 2],
#     #     mode='markers',
#     #     marker=dict(
#     #         size=2,
#     #         # color=pred,
#     #         # color=ncoord2[:, 3],
#     #         # colorscale='Viridis',
#     #         opacity=0.8
#     #     )
#     # )
#     # fig = make_subplots(
#     #     rows=1, cols=1,
#     #     specs=[[{'type': 'surface'}]])
#     # fig.add_trace(subfig)
#     # fig.update_layout(scene_aspectmode='data', )
#     # fig.show()


def get_hidden_leaf_characteristics(leaf):
    # Copié d'autre endroit, permet de récupérer le bout de la feuille
    tempLeaf = np.copy(leaf)
    distFromCenter = np.sqrt(tempLeaf[:, 0] ** 2 + tempLeaf[:, 1] ** 2)
    tempLeaf = np.append(np.arange(len(tempLeaf))[np.newaxis].T, tempLeaf, axis=1)
    tempLeaf = np.append(tempLeaf, distFromCenter[np.newaxis].T, axis=1)
    tempLeaf = tempLeaf[tempLeaf[:, 4].argsort()]

    pca = PCA(n_components=3)
    leafPCA = pca.fit_transform(X=leaf)
    eigenVectorX, eigenVectorY = pca.components_[0], pca.components_[1]
    planNormal = np.cross(eigenVectorX, eigenVectorY)
    if planNormal[2] < 0:
        planNormal *= -1
    angle = math.acos(np.dot(planNormal, [0, 0, 1])/(np.linalg.norm(planNormal)))
    angle = math.pi/2 - angle
    angle *= 180/math.pi
    d = np.dot(planNormal, tempLeaf[-1, 1:4])  # Equation of the plan, planNormal = [a, b, c]
    zNode = d  # Because the stem is a segment of the z axis, intersection with plan and stem is d
    length = np.linalg.norm(tempLeaf[-1, 1:4] - [0, 0, d])

    yLeaf = np.copy(leafPCA)
    yLeaf = np.append(np.arange(len(yLeaf))[np.newaxis].T, yLeaf, axis=1)
    yLeaf = yLeaf[yLeaf[:, 2].argsort()]
    width = yLeaf[-1, 2] - yLeaf[0, 2]

    # subfig = go.Scatter3d(
    #     # x=leafPCA[:, 0],
    #     # y=leafPCA[:, 1],
    #     # z=leafPCA[:, 2],
    #     x=leafPCA[:, 0],
    #     y=leafPCA[:, 1],
    #     z=leafPCA[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=2,
    #         # color=pred,
    #         # color=ncoord2[:, 3],
    #         # color=templist,
    #         # colorscale='Viridis',
    #         opacity=0.8
    #     )
    # )
    # fig = make_subplots(
    #     rows=1, cols=1,
    #     specs=[[{'type': 'surface'}]])
    # fig.add_trace(subfig)
    # fig.update_layout(scene_aspectmode='data', )
    # fig.show()
    return length, width, angle, zNode


def separate_united_leaves(clusters):  # list of clusters with removed clusters set as the following list : [1]
    listCenters = list()
    for cluster in clusters:
        if len(cluster) != 1:
            x, y, z = getCenterCluster(cluster)
            listCenters.append(np.array([x, y, z]))
    counter = 0
    indexOfWrongClusters = list()
    for i in range(len(listCenters)):
        dist = np.linalg.norm(listCenters[i][0:2])
        if dist < 0.3:
            counter += 1
            indexOfWrongClusters.append(i)

    for i in indexOfWrongClusters:
        pca = PCA(n_components=3)
        sortedCluster = pca.fit_transform(X=clusters[i])
        sortedCluster = np.append(np.arange(len(sortedCluster))[np.newaxis].T, sortedCluster, axis=1)
        sortedCluster = sortedCluster[sortedCluster[:, 1].argsort()]
        xCenter = (sortedCluster[-1, 1] + sortedCluster[0, 1])/2

        leaf1, leaf2 = split(sortedCluster[:, 1:], sortedCluster[:, 1] < xCenter)  # We do not keep the index column

        # Retrieve initial data of the leaf (removing PCA transformation, ...)
        leaf1 = leaf1[leaf1[:, 0].argsort()]  # Removing the previous sorting
        leaf1 = pca.inverse_transform(leaf1)  # Removing PCA transform

        leaf2 = leaf2[leaf2[:, 0].argsort()]  # Removing the previous sorting
        leaf2 = pca.inverse_transform(leaf2)  # Removing PCA transform

        clusters[i] = leaf1
        clusters.append(leaf2)
    return clusters


def read_data(dt):
    # Reading the txt file
    f = open("../BasilData/basil_coordinates_t{}.txt".format(dt), "r")
    # f = open("basil_coordinates_feuilles_plates.txt", "r")
    coord = f.read()
    coord = coord.split('\n')
    ncoord = np.zeros((len(coord), 3))
    for i in range(len(coord) - 1):
        coord[i] = coord[i].split(',')
        for j in range(3):
            ncoord[i][j] = float(coord[i][j])
    f.close()
    return ncoord


def plot_data(coord, label):
    subfig = go.Scatter3d(
        x=coord[:, 0],
        y=coord[:, 1],
        z=coord[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=label,
            colorscale='Viridis',
            opacity=0.8
        )
    )
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'surface'}]])
    fig.add_trace(subfig)
    fig.update_layout(scene_aspectmode='data', )
    fig.show()


def get_numpy_array(leaves):
    ncoord = leaves[0]
    ncoord = np.append(ncoord, np.ones((len(leaves[0]), 1)) * 0, axis=1)
    for i in range(1, len(leaves)):
        if len(leaves[i]) > 1:
            temp = np.append(leaves[i], np.ones((len(leaves[i]), 1)) * i, axis=1)
            ncoord = np.append(ncoord, temp, axis=0)
    return ncoord


# Just junk codes to test the functions
# for blabla in range(46, 48):
#     # en t (ou blabla) = 38, le centre de gravité est loin du centre donc y a pas de séparation de feuille :'(
#     # 18
#     # f = open("basil_coordinates2.txt", "r")
#     ncoord = read_data(blabla)
#
#     plantHeight = get_height(ncoord)
#
#     length, width, height, trueLength = 1280, 720, 50, 50
#
#     dp = trueLength/length  # the length of a pixel if it hits the ground
#
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # ax.scatter(ncoord[:,0], ncoord[:,1], ncoord[:,2], s=300)
#     # ax.view_init(azim=200)
#     # plt.show()
#
#     # get_leaves(ncoord)
#
#     leaves, nb_classes, pred = get_leaves(ncoord)
#
#     plot_data(ncoord, pred)
#     if nb_classes > 1:
#         un, deux = getUpperLeaves(leaves)
#         leaves = separate_united_leaves(leaves)
#         ncoord2 = np.copy(leaves[0])
#         ncoord2 = np.append(ncoord2, np.ones((len(leaves[0]), 1))*0, axis=1)
#         for j in range(1, len(leaves)):
#             if len(leaves[j]) != 1:
#                 temp = np.copy(leaves[j])
#                 # Il faut rajouter une colonne qui est le label
#                 temp = np.append(temp, np.ones((len(leaves[j]), 1))*j, axis=1)
#                 ncoord2 = np.append(ncoord2, temp, axis=0)
#         plot_data(ncoord2[:, :3], ncoord2[:, 3])
#
#
#     # model = DBSCAN(eps=4*dp, min_samples=15, n_jobs=4)
#     # model.fit_predict(ncoord)
#     # pred = model.fit_predict(ncoord)
#     #
#     # nb_classes = np.max(pred) + 1  # The label are going from 0 to n-1, so n is the number of label and is the max
#     # # value possible for a label
#     # if nb_classes > 1:
#     #     leaves = [[]]*nb_classes
#     #     li = []
#     #     compteur = 0
#     #     for i in range(len(ncoord)):
#     #         if pred[i] != -1:
#     #             if len(leaves[pred[i]]) == 0:
#     #                 leaves[pred[i]] = [ncoord[i, :]]
#     #             else:
#     #                 leaves[pred[i]].append(ncoord[i, :])
#     #
#     #     centers = list()
#     #     for i in range(nb_classes):
#     #         leaves[i] = np.array(leaves[i])
#     #         centers.append(np.abs(np.sum(leaves[i][:, 0]) / len(leaves[i][:, 0]) + np.sum(leaves[i][:, 1]) / len(leaves[i][:, 1])))
#     #
#     #     indexOfUpperLeaves = centers.index(min(centers))
#     #     # Another technique probably better is to count the number of point with x positive and with x negative, if it is the same, it's ok (idem with y)
#     #
#     #     # uniteClusters(leaves, 1, nb_classes)
#     #
#     #     leaves2 = []
#     #     for i in range(len(leaves)):  # delete cluster without enough point
#     #         # if len(leaves[i]) > 0.01 * len(coord):
#     #         #     leaves2.append(leaves[i])
#     #         # if len(leaves[i]) < 0.01 * len(coord):
#     #         if len(leaves[i]) < 100:
#     #             leaves[i] = np.array([1])  # Just a definite array to check whether we keep the cluster or not
#     #     leaves2 = leaves
#     #     nb_points_clean = 0
#     #     for cluster in leaves2:
#     #         if len(cluster) != 1:
#     #             nb_points_clean += len(cluster)
#     #     # ncoord2 = np.zeros((nb_points_clean, 3))
#     #     ncoord2 = np.zeros((nb_points_clean, 4))
#     #     leaves2 = separate_united_leaves(leaves2)
#     #     indice = 0
#     #     for i in range(len(leaves2)):
#     #         if len(leaves2[i]) != 1:
#     #             ncoord2[indice:indice+len(leaves2[i]), :-1] = leaves2[i]
#     #             ncoord2[indice:indice + len(leaves2[i]), -1] = np.full((len(leaves2[i])), i)
#     #             indice += len(leaves2[i])
#     #
#     #     un, deux = getUpperLeaves(leaves2)
#     #
#     #     # length, width, leafPCA, angle2, zNode = get_parameters(leaves2[int(un)])
#     #
#     #     # xmean, ymean, _ = getCenterCluster(leaves2[int(un)])
#     #     # planNormal = [xmean, ymean, 0]
#     #     # angle8, zNode8, _, _ = get_hidden_leaf_characteristics(leaves2[8])
#     #     # angle0, zNode0, _, _ = get_hidden_leaf_characteristics(leaves2[0])
#     #     # get_hidden_leaf_characteristics(leaves2[2])
#     #
#     #     subfig = go.Scatter3d(
#     #         # x=leaves[2][:, 0],
#     #         # y=leaves[2][:, 1],
#     #         # z=leaves[2][:, 2],
#     #         # x=ncoord[:, 0],
#     #         # y=ncoord[:, 1],
#     #         # z=ncoord[:, 2],
#     #         x=ncoord2[:, 0],
#     #         y=ncoord2[:, 1],
#     #         z=ncoord2[:, 2],
#     #         mode='markers',
#     #         marker=dict(
#     #             size=2,
#     #             # color=pred,
#     #             color=ncoord2[:, 3],
#     #             colorscale='Viridis',
#     #             opacity=0.8
#     #         )
#     #     )
#     #     fig = make_subplots(
#     #         rows=1, cols=1,
#     #         specs=[[{'type': 'surface'}]])
#     #     fig.add_trace(subfig)
#     #     fig.update_layout(scene_aspectmode='data', )
#     #     fig.show()
#     #
#     #     # subfig = go.Scatter3d(
#     #     #     # x=leaves[2][:, 0],
#     #     #     # y=leaves[2][:, 1],
#     #     #     # z=leaves[2][:, 2],
#     #     #     x=ncoord[:, 0],
#     #     #     y=ncoord[:, 1],
#     #     #     z=ncoord[:, 2],
#     #     #     # x=ncoord2[:, 0],
#     #     #     # y=ncoord2[:, 1],
#     #     #     # z=ncoord2[:, 2],
#     #     #     mode='markers',
#     #     #     marker=dict(
#     #     #         size=2,
#     #     #         color=pred,
#     #     #         # color=ncoord2[:, 3],
#     #     #         colorscale='Viridis',
#     #     #         opacity=0.8
#     #     #     )
#     #     # )
#     #     # fig = make_subplots(
#     #     #     rows=1, cols=1,
#     #     #     specs=[[{'type': 'surface'}]])
#     #     # fig.add_trace(subfig)
#     #     # fig.update_layout(scene_aspectmode='data', )
#     #     # fig.show()
#     #
#     #     # popo = 2  # 0 feuille partielle, 8 feuille partielle, 2 lune
#     #     # tempLeaf = np.copy(leaves2[popo])
#     #     # distFromCenter = np.sqrt(tempLeaf[:, 0] ** 2 + tempLeaf[:, 1] ** 2)
#     #     # tempLeaf = np.append(np.arange(len(tempLeaf))[np.newaxis].T, tempLeaf, axis=1)
#     #     # tempLeaf = np.append(tempLeaf, distFromCenter[np.newaxis].T, axis=1)
#     #     # tempLeaf = tempLeaf[tempLeaf[:, 4].argsort()]
#     #     # templist = np.zeros(len(leaves2[popo]))
#     #     # templist[int(tempLeaf[-1, 0])] = 1
#     #     # subfig = go.Scatter3d(
#     #     #     # x=leafPCA[:, 0],
#     #     #     # y=leafPCA[:, 1],
#     #     #     # z=leafPCA[:, 2],
#     #     #     x=leaves2[popo][:, 0],
#     #     #     y=leaves2[popo][:, 1],
#     #     #     z=leaves2[popo][:, 2],
#     #     #     mode='markers',
#     #     #     marker=dict(
#     #     #         size=2,
#     #     #         # color=pred,
#     #     #         # color=ncoord2[:, 3],
#     #     #         color=templist,
#     #     #         colorscale='Viridis',
#     #     #         opacity=0.8
#     #     #     )
#     #     # )
#     #     # fig = make_subplots(
#     #     #     rows=1, cols=1,
#     #     #     specs=[[{'type': 'surface'}]])
#     #     # fig.add_trace(subfig)
#     #     # fig.update_layout(scene_aspectmode='data', )
#     #     # fig.show()
#     #
#     #     # fig = plt.figure()
#     #     # ax = Axes3D(fig)
#     #     # ax.scatter(ncoord[:,0], ncoord[:,1], ncoord[:,2], c=model.labels_, s=300)
#     #     # ax.view_init(azim=200)
#     #     # plt.show()
#     #
#     #     print("number of cluster found: {}".format(len(set(model.labels_))))
#     #     print('cluster for each point: ', model.labels_)