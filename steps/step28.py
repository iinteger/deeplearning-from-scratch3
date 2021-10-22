import numpy as np
from dezero import Variable


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 1000

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

if __name__ == "__main__":
    y = rosenbrock(x0, x1)
    y.backward()
    print(x0.grad, x1.grad)
    # (x0, x1 : -2.0, 400) : 함수의 출력을 가장 크게 하는 방향