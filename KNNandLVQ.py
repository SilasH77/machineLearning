from random import randrange

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
# import some data
iris = datasets.load_iris()
#retrieve the data
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)


def getEuclideanDistance(current_point, test_point):
    distance = 0
    # loop through all features at the current point
    for feature in range(len(current_point)):
        # calculate euclidean distance for current feature between test point and current training point
        distance = distance + (current_point[feature] - test_point[feature]) ** 2

    # get square root of distance
    distance = np.sqrt(distance)
    return distance

def getDistances(x_train, test_point):
    #this array will hold the distances from our test point to other points
    distances = []

    # get distance from a test point to all other points
    # loop through all points in the training set of features
    for instance in range(len(x_train)):
        #set current point to point in training set
        current_point = x_train[instance]

        distance = getEuclideanDistance(current_point,test_point)

        #add distance calculation to distance vector
        current = [instance, current_point[0], current_point[1], current_point[2], current_point[3], distance]
        distances.append(current)

    return distances
def classifyKNN(x_train, test_point, k):

    distances = getDistances(x_train, test_point)
    #STEP 2 get KNeighbors
    distances.sort(key = lambda x: x[5])
    nearest = distances[:k]

    #STEP 3 classify based on most common
    classificationResults = [0 for x in range(len(nearest))]
    for x in range(len(nearest)):
        classificationResults[y_train[nearest[x][0]]] += 1
    return (classificationResults.index(max(classificationResults)))


def createCodebooks(x_train, y_train, k):
    # create codebook for each class (0,1,2)
    instance = len(x_train)
    features = len(x_train[0])
    codebooks = []
    for i in range(k):
        randomLocation = randrange(instance)
        codebook = [x_train[randomLocation][i] for i in range(features)]
        codebook.append(y_train[randomLocation])
        codebooks.append(codebook)

    return codebooks

def trainCodebooks(x_train, y_train, codebooks, epochs, alpha):
    for x in range(epochs):
        currentRate = alpha * (1.0 - (x / epochs))
        #loop through all points in the training set of features
        for instance in range(len(x_train)):
            codebookDistances = []
            #loop through each codebook
            for codebook in codebooks:
                #add the distance from the current point to codebook distances
                codebookDistances.append(getEuclideanDistance(x_train[instance], codebook))

            #if the class of the closest codebook matches the test point, move it closer
            location = codebookDistances.index(min(codebookDistances))
            if codebooks[location][4] == y_train[instance]:
                #move closer
                for feature in range(len(x_train[instance])):
                    codebooks[location][feature] = (codebooks[location][feature] - x_train[instance][feature])
                    codebooks[location][feature] = codebooks[location][feature] * currentRate
                    codebooks[location][feature] = (x_train[instance][feature] + codebooks[location][feature])
            else:
                # move further
                for feature in range(len(x_train[instance])):
                    codebooks[location][feature] = (codebooks[location][feature] - x_train[instance][feature])
                    codebooks[location][feature] = codebooks[location][feature] * currentRate
                    codebooks[location][feature] = (x_train[instance][feature] - codebooks[location][feature])

    return codebooks

def LVQ(x_train, y_train, test_point):

    #create a codebook vector for each class, starting at a random location
    codebooks = createCodebooks(x_train,y_train,3)
    codebooks2 = trainCodebooks(x_train,y_train, codebooks, 200, 0.3)
    # loop through each codebook
    codebookDistances = []
    for codebook in codebooks2:
        # add the distance from the current point to codebook distances
        codebookDistances.append(getEuclideanDistance(test_point, codebook))

    # if the class of the closest codebook matches the test point, move it closer
    location = codebookDistances.index(min(codebookDistances))
    print("selected match: " + str(codebooks[location][4]))







print(classifyKNN(x_train, x_train[2],3))
print("proposed match: " + str(y_train[5]))
LVQ(x_train, y_train, x_train[5])
