from perceptron import Perceptron
from perceptron import Point
import matplotlib.pyplot as pyplot
import ast
import random
import numpy

pointCount = 400
intersectionCount = 0
inputNodeCount = 2
hiddenNodeCount = 100
outputNodeCount = 30
layerCount = 3
learningRate = 0.2
outputMode = 'n'

perc = Perceptron(inputNodeCount,  # Число входных нейронов (признаков)
                  hiddenNodeCount, # Число скрытых (промежуточных) нейронов
                  outputNodeCount, # Число выходных нейронов
                  layerCount,      # Число слоев
                  learningRate,
                  outputMode) 


def readDataFromFile(fileName):
    points = []
    print('START READING POINTS FROM FILE')
    text_file = open(fileName, "r")
    index = 0
    for line in text_file:
        lineArr = ast.literal_eval(line)
        for point in lineArr:
            points.append(Point(index,point))
        index += 1
    print('FINISHED READING POINTS FROM FILE')
    text_file.close()
    return points

def randomColor(): # Задание случайного цвета для отображения множества
    colors = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    colorArr = []
    for j in range(30):  
        color = '#'
        for i in range(6):
            color += colors[random.randint(0, 15)]
        colorArr.append(color)
    return colorArr

def visualize(points): # Метод для визуализации множеств
    colorArr = randomColor()
    for i in range(points.__len__()):
        x = []
        y = []
        for j in range(points[i].__len__()):
            x.append(points[i][j][0])
            y.append(points[i][j][1])
        pyplot.scatter(x,y,c=colorArr[i],alpha=0.8,s=10)
    pyplot.rcParams['axes.facecolor']='grey'
    pyplot.gca().set_xlim([0, 1])
    pyplot.gca().set_ylim([0, 1])
    pyplot.show()
    
def flattenArray(points):
    flatArray = []
    for cl in points:
        flatArray += cl
    return flatArray

outputPoints = []
for i in range(outputNodeCount):
    outputPoints.append([])

def train(points):
    for i in range(int(len(points)/2)):
        out = numpy.zeros(outputNodeCount)
        out[points[i].classNum] = 1
        perc.train(points[i].vars, out)
    print('CORRECT GUESSES: ', perc.correct_guesses_T, ' OUT OF ', len(points)/2, '; ', perc.correct_guesses_T / (len(points)/2))

def validate(points):
    half = int(len(points)/2)
    for i in range(half, half * 2):
        out = numpy.zeros(outputNodeCount)
        out[points[i].classNum] = 1
        outputPoint = perc.query(points[i].vars, out)
        outputPoints[outputPoint.classNum].append(outputPoint.vars)
    print('CORRECT GUESSES: ', perc.correct_guesses_V, ' OUT OF ', len(points)/2, '; ', perc.correct_guesses_V / (len(points)/2))
    
points = readDataFromFile('generated_sets/output_p{0}_cl{1}_var{2}_int{3}.txt'.format(pointCount, outputNodeCount, inputNodeCount, intersectionCount))
random.shuffle(points)

train(points)
validate(points)
#
#pyplot.scatter(perc.graph_x_T, perc.graph_y_T, s=1)
#pyplot.show()
#
#pyplot.scatter(perc.graph_x_V, perc.graph_y_V, s=1)
#pyplot.show()
visualize(outputPoints)