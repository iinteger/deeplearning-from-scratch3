from step03 import *


# 중앙 차분
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2*eps)


if __name__ == '__main__':
    f = Square()
    x = Variable(np.array(2.0))
    dy =numerical_diff(f, x)
    print(dy)