from step01 import *


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)  # 구체적인 계산은 forward에서 함
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()  # 상속해서 구현해야 함


class Square(Function):
    def forward(self, x):
        return x ** 2