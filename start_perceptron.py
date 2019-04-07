from perceptron import Perceptron
from perceptron import Point
import matplotlib.pyplot as pyplot
import ast
import random
import numpy

pointCount = 300
intersectionCount = 0
inputNodeCount = 2
hiddenNodeCount = 100
outputNodeCount = 20
layerCount = 3
learningRate = 0.3
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

def flattenArray(points):
    flatArray = []
    for cl in points:
        flatArray += cl
    return flatArray

def train(points):
    for i in range(int(len(points)/2)):
        out = numpy.zeros(outputNodeCount)
        out[points[i].classNum] = 1
        perc.train(points[i].vars, out)
    #print('CORRECT GUESSES: ', perc.correct_guesses, ' OUT OF ', len(points)/2, '; ', perc.correct_guesses / (len(points)/2))

def validate(points):
    half = int(len(points)/2)
    for i in range(half, half * 2):
        out = numpy.zeros(outputNodeCount)
        out[points[i].classNum] = 1
        perc.query(points[i].vars, out)
    print('CORRECT GUESSES: ', perc.correct_guesses, ' OUT OF ', len(points)/2, '; ', perc.correct_guesses / (len(points)/2))
    
points = readDataFromFile('generated_sets/output_p{0}_cl{1}_var{2}_int{3}.txt'.format(pointCount, outputNodeCount, inputNodeCount, intersectionCount))
random.shuffle(points)

train(points)
validate(points)

pyplot.scatter(perc.graph_x, perc.graph_y, s=1)
#pyplot.gca().set_xlim([0, len(perc.graph_x)])
#pyplot.gca().set_ylim([0, 1])
pyplot.show()
#flatArray = flattenArray(points)
#print(flatArray)