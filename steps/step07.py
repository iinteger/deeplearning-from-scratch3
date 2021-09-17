import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None  # data와 대응하는 미분값
        self.creator = None

    def set_creator(self, func):  # 변수의 관점에서 함수는 자신을 만들어낸 창조자(creator)
        self.creator = func

    def backward(self):  # 재귀를 사용해 끝까지 역전파 계산
        f = self.creator
        if f is not None:  # f가 None일땐 함수 바깥에서 생성된 변수를 의미
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)  # 계산이 끝난 결과를 클래스에 담아서 return
        output.set_creator(self)  # 이 함수가 변수의 창조자임을 저장. 자동 역전파에서 이를 참조
        self.input = input  # 입력 변수 기억. backward에서 재사용
        self.output = output  # 출력 저장
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


if __name__ == '__main__':
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)