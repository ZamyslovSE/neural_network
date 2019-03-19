from perceptron import Perceptron
from perceptron import Point
import ast
import random
import numpy

inputNodeCount = 2
hiddenNodeCount = 60
outputNodeCount = 5
layerCount = 3
learningRate = 0.3

perc = Perceptron(inputNodeCount,  # Число входных нейронов (признаков)
                  hiddenNodeCount, # Число скрытых (промежуточных) нейронов
                  outputNodeCount, # Число выходных нейронов
                  layerCount,      # Число слоев
                  learningRate)    # 


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

points = readDataFromFile('generated_sets/output_p300_cl5_var2_int0.txt')
random.shuffle(points)

train(points)
#flatArray = flattenArray(points)
#print(flatArray)