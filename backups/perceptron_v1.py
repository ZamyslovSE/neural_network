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
        
        self.inputNodeCount = inputNodeCount
        self.hiddenNodeCount = hiddenNodeCount
        self.outputNodeCount = outputNodeCount
        
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
        # преобразование списка входных значений
        # в двухмерный массив
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.Wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.Who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        print('TRAINING. OUTPUT:\n {0}\n EXPECTED OUTPUT:\n {1}'.format(final_outputs, targets))
        # ошибки выходного слоя =
        # (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.Who.T, output_errors)
        # обновить весовые коэффициенты для связей между
        # скрытым и выходным слоями
        self.Who += self.learningRate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # обновить весовые коэффициенты для связей между
        # входным и скрытым слоями
        self.Wih += self.learningRate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
      
    # Запрос к нейронной сети
    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        
        start_inputs = numpy.dot(self.Wih, inputs)
        start_outputs = self.activation_function(start_inputs)
        
        final_inputs = numpy.dot(self.Who, start_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs