import matplotlib.pyplot as pyplot
import random
import numpy as math

points = [] # Массив точек
centers = [] # Массив центров классов

pointCount = 200 # Количество точек в классе
classCount = 10   # Количество классов
varCount = 2     # Количество признаков
radius = 0.05     # Радиус разброса
intersectionCount = 5 # Количество пересечений
intersectionArea = 0.25
approximation = 0.05
intersectionsLeft = intersectionCount

def generateClass(pointCount, varCount, centerPoint, radius):
    points = []
    for i in range(0, pointCount):
        points.append([])
        maxVal = radius**2
        for j in range (0, varCount):
            p = random.uniform(-math.sqrt(maxVal), math.sqrt(maxVal))
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
    
def checkIntersectionArea(d, r):
    x1 = 2*math.arcsin(math.sqrt(r**2 - (d**2)/4)/r)
    x2 = math.sin(2*math.arcsin(math.sqrt(r**2 - (d**2)/4)/r))
    return (x1 - x2)/math.pi
    
def countPointDistance(point1,point2):
    dist = 0
    for i in range(len(point1)):
        dist += (point2[i]-point1[i])**2
    return math.sqrt(dist)

def checkIntersectionCount(centers, point, radius): # TRUE когда пересекаются
    count = 0
    for center in centers:
        if (countPointDistance(center,point) < radius*2):
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
                flag = True
                for center in centers:
                    area = checkIntersectionArea(countPointDistance(center,newCenter),radius)
                    if (area > 0):
                        if not (area > intersectionArea - approximation and area < intersectionArea + approximation):
                            flag = False
                if (flag):        
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

visualize(generatePoints(pointCount, classCount, varCount, radius, intersectionCount))
