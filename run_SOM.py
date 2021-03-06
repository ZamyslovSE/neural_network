from SOM import SOM
from SOM import Neuron
import matplotlib.pyplot as pyplot
import random
import numpy
import ast

varCount = 2
gridCount = 80
learningRate = 0.1
searchRadius = 0.4
tau1 = 1000
tau2 = 1000
pointCount = 1500

#INIT NEURON MATRIX
ar = numpy.linspace(0.0, 1.0, gridCount)
neurons = []
for i in range(gridCount):
    for j in range(gridCount):
        neurons.append(Neuron([ar[i], ar[j]],[i,j]))
        
som = SOM(neurons, varCount, learningRate)

def randomColor(classCount): # Задание случайного цвета для отображения множества
    vals = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    colorArr = []
    for j in range(classCount):  
        color = '#'
        for i in range(6):
            color += vals[random.randint(0, 15)]
        colorArr.append(color)
    return colorArr

def flattenArray(points):
    flatArray = []
    for cl in points:
        flatArray += cl
    return flatArray

def read(fileName):
    points = []
    print('START READING POINTS FROM FILE')
    text_file = open(fileName, "r")
    for line in text_file:
        lineArr = ast.literal_eval(line)
        points.append(lineArr)
    print('FINISHED READING POINTS FROM FILE')
    text_file.close()
    return points

def visualize(points, neurons): # Метод для визуализации множеств
    colorArr = randomColor(1)
    colorArr.append('#000000')
    x = []
    y = []
    for j in range(points.__len__()):
        x.append(points[j][0])
        y.append(points[j][1])
    pyplot.scatter(x,y,c=colorArr[0],alpha=0.4,s=5, marker="o")
    x = []
    y = []
    for j in range(neurons.__len__()):
        x.append(neurons[j].W[0])
        y.append(neurons[j].W[1])
    pyplot.scatter(x,y,c=colorArr[1],alpha=0.4,s=5, marker="s")
    pyplot.gca().set_xlim([0, 1])
    pyplot.gca().set_ylim([0, 1])
    pyplot.gca().set_aspect('equal', adjustable='box')
    pyplot.show()
    

    
points = read('generated_sets/output_p300_cl10_var2_int0.txt')
#visualize(points)
pointsFlat = flattenArray(points)
random.shuffle(pointsFlat)
som.train(pointsFlat[0:pointCount], 5, searchRadius, tau1, tau2)
visualize(pointsFlat[0:pointCount], som.neurons)