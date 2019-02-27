import math
import numpy as np


def softmax(x) -> np.ndarray:
    return np.exp(x) / sum(np.exp(x))


def cost_fun(x, y) -> float:
    return sum(x * math.log(y))


class Layer:
    def __init__(self, depth, inSize, layerSize, outSize):
        depth -= 1
        self.w = np.random.rand(layerSize, inSize) / layerSize
        self.b = (np.random.rand(layerSize, 1) / layerSize)[:, 0]
        self.inputs = inSize
        self.outputs = layerSize
        if depth > 0:
            self.nextLayer = Layer(depth, layerSize, layerSize, outSize)
        else:
            self.nextLayer = OutLayer(layerSize, outSize)

    def teach(self, x, y, learnR):
#        print(self.w.shape)
#        print(self.b.shape)
        inVec = np.dot(self.w, x) + self.b
        output = np.tanh(inVec)
        nextDiff = self.nextLayer.teach(output, y, learnR)
        diff = nextDiff * (1/(np.exp(inVec) + np.exp(-inVec)))
        prevDiff = np.dot(self.w.T, diff)
        gw = np.matmul(diff.reshape(self.outputs, 1), x.reshape(1, self.inputs))
        gb = diff
        self.w -= learnR * gw
        self.b -= learnR * gb
        return prevDiff


class OutLayer:
    def __init__(self, layerSize, outSize):
        self.w = np.random.rand(outSize, layerSize) / outSize
        self.b = (np.random.rand(outSize, 1) / outSize)[:, 0]
        self.inputs = layerSize
        self.outputs = outSize

    def teach(self, x, y, learnR):
        inVec = np.dot(self.w, x) + self.b
        output = softmax(inVec)
        diff = output - y
        prevDiff = np.dot(self.w.T, diff)
        gw = np.matmul(diff.reshape(self.outputs, 1), x.reshape(1, self.inputs))
        gb = diff
        self.w -= learnR * gw
        self.b -= learnR * gb
        return prevDiff
