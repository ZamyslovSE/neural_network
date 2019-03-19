import matplotlib.pyplot as pyplot
import random
import numpy

points = [] # Массив точек
centers = [] # Массив центров классов

pointCount = 200 # Количество точек в классе
classCount = 30   # Количество классов
varCount = 2     # Количество признаков
radius = 0.05     # Радиус разброса
intersectionCount = 10 # Количество пересечений
intersectionsLeft = intersectionCount

def generateClass(pointCount, varCount, centerPoint, radius):
    points = []
    for i in range(0, pointCount):
        points.append([])
        maxVal = radius**2
        for j in range (0, varCount):
            p = random.uniform(-numpy.sqrt(maxVal), numpy.sqrt(maxVal))
            points[i].append(centerPoint[j] + p)
            maxVal = maxVal - p**2
    return points

def randomColor():
    colorArr = []
    for j in range(30):  
        color = '#'
        for i in range(6):
            color += str(random.randint(0, 9))
        colorArr.append(color)
    return colorArr

def visualize(points):
    colorArr = randomColor()
    for i in range(points.__len__()):
        x = []
        y = []
        for j in range(points[i].__len__()):
            x.append(points[i][j][0])
            y.append(points[i][j][1])
        pyplot.scatter(x,y,c=colorArr[i],alpha=0.3,s=1)
    pyplot.gca().set_xlim([0, 1])
    pyplot.gca().set_ylim([0, 1])
    pyplot.show()
    
def checkIntersectionCount(centers, point, radius): # TRUE когда пересекаются
    count = 0
    max = (radius*2)
    for center in centers:
        dist = 0
        for i in range(len(point)):
            dist += (center[i]-point[i])**2
        if (numpy.sqrt(dist) < max):
            count += 1
    return count

def generateCenter(centers, varCount, radius, intersectionsLeft):
    if (len(centers) == 0): # Первый центр ставится случайным образом
        newCenter = []
        for j in range(0, varCount):
            newCenter.append(random.uniform(0.1,0.9))
        print('generate first center: ',newCenter)
        return newCenter
    
    while True:
        newCenter = []
        for j in range(0, varCount):
            newCenter.append(random.uniform(0.1,0.9))
        intCount = checkIntersectionCount(centers, newCenter, radius)
        if (intersectionsLeft > 0):
            if (intCount > 0 and intCount <= intersectionsLeft):
                intersectionsLeft -= intCount
                print('generate center: {0} , int={1}, left={2}'.format(newCenter,intCount,intersectionsLeft))
                return newCenter
        if (intersectionsLeft == 0):
            if (intCount == 0):
                print('generate center: {0} , int={1}, left={2}'.format(newCenter,intCount,intersectionsLeft))
                return newCenter

def generatePoints(pointCount, classCount, varCount, radius, intersectionCount):
    points = []
    centers = []
    intersectionsLeft = intersectionCount
    for i in range(0,classCount):
        newCenter = generateCenter(centers, varCount, radius, intersectionsLeft)
        intersectionsLeft -= checkIntersectionCount(centers, newCenter, radius)
        centers.append(newCenter)
        points.append(generateClass(pointCount, varCount, newCenter, radius))
    return points

print(checkIntersectionCount([[0.2,0.2]],[0.4,0.4],0.1))

visualize(generatePoints(pointCount, classCount, varCount, radius, intersectionCount))

