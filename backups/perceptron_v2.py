import numpy
import scipy.special


class Point:
    # Конструктор
    def __init__(self, classNum, vars):        
        self.classNum = classNum      
        self.vars = vars
        
    def __str__(self):
        return f'[{self.classNum}, {self.vars}]'
    
    def __repr__(self):
        return f'[{self.classNum}, {self.vars}]'
class Perceptron:
    # Конструктор
    def __init__(self, inputNodeCount, hiddenNodeCount, outputNodeCount, layerCount, learningRate):
        print('START INIT PERCEPTRON')
        print('PARAMETERS: inputs={0} hidden={1} output={2} layerCount={3}'.format(inputNodeCount, hiddenNodeCount, outputNodeCount, layerCount))

        
        self.inputNodeCount = inputNodeCount
        self.hiddenNodeCount = hiddenNodeCount
        self.outputNodeCount = outputNodeCount
        
        self.layerCount = layerCount
        
        self.learningRate = learningRate
        
        self.Wih = (numpy.random.rand(self.hiddenNodeCount, self.inputNodeCount) - 0.5)
        self.Whh = []
        for i in range(layerCount-1):
            self.Whh.append(numpy.random.rand(self.hiddenNodeCount, self.hiddenNodeCount) - 0.5)
        self.Who = (numpy.random.rand(self.outputNodeCount, self.hiddenNodeCount) - 0.5)
        
        self.activation_function = lambda x: scipy.special.expit(x)
        
        print('FINISH INIT PERCEPTRON')
        
    # Обучение нейронной сети
    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        
        W = [self.Wih]
        for Wi in self.Whh:
            W.append(Wi)
        W.append(self.Who)
        outputs = [inputs]
        
        #start_inputs = numpy.dot(self.Wih, inputs)
        #start_outputs = self.activation_function(start_inputs)
        #outputs.append(start_outputs)
        
        for i in range(len(W)):
            hidden_inputs = numpy.dot(W[i], outputs[i])
            hidden_outputs = self.activation_function(hidden_inputs)
            outputs.append(hidden_outputs)
        
        final_outputs = outputs[len(outputs)-1]
        
        print('TRAINING. OUTPUT:\n {0}\n EXPECTED OUTPUT:\n {1}'.format(final_outputs, targets))
        
        output_errors = targets - final_outputs
        errors = [output_errors]
        index = 0
        for i in reversed(range(len(W))):
            hidden_errors = numpy.dot(W[i].T, errors[0])
            W[i] += self.learningRate * numpy.dot((errors[0] * outputs[i+1] * (1.0 - outputs[i+1])), numpy.transpose(outputs[i]))
            errors.insert(0, hidden_errors)
            #self.Who += self.learningRate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(start_outputs))
            #self.Wih += self.learningRate * numpy.dot((hidden_errors * start_outputs * (1.0 - start_outputs)), numpy.transpose(inputs))
            
        self.Wih = W[0]
        for i in range(self.layerCount-2):
            self.Whh[i] = W[i+1]
        self.Who = W[self.layerCount]
        pass
      
    # Запрос к нейронной сети
    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        
        start_inputs = numpy.dot(self.Wih, inputs)
        start_outputs = self.activation_function(start_inputs)
        
        final_inputs = numpy.dot(self.Who, start_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs