import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]  # 입력을 가변 길이 인수로 변경
        ys = self.forward(*xs)  # 애스터리스크로 언팩
        if not isinstance(ys, tuple):  # 튜플이 아니면 튜플로 묶어줌
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):  # 입력받은 데이터가 ndarray가 아니면 에러 발생
                raise TypeError("{}는 지원하지 않습니다.".format(type(data)))

        self.data = data
        self.grad = None  # data와 대응하는 미분값
        self.creator = None

    def set_creator(self, func):  # 변수의 관점에서 함수는 자신을 만들어낸 창조자(creator)
        self.creator = func

    def backward(self):  # 반복문을 사용해 구현.
        if self.grad is None:  # 기울기가 없을때 자동으로 초기 미분값 생성
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]  # 출력변수 outputs에 담긴 미분값들을 리스트에 담음
            gxs = f.backward(*gys)  # 역전파
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):  # 역전파로 흘려보내는 미분값을 Variable의 grad 필드에 담음
                if x.grad is None:  # 미분값을 처음 설정할 때는 출력에서 전달되는 미분값을 그대로 사용해야함
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):  # 같은 변수를 다른 계산에 사용할 때 문제가 생기기 때문에 기울기 초기화 필요
        self.grad = None

if __name__ == "__main__":
    x = Variable(np.array(3.0))
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)
