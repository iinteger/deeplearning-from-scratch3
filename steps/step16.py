# 역전파를 할 때 사용된 함수를 리스트에서 꺼내오는데, 병렬로 이어진 노드는 올바른 순서대로 역전파가 처리되지 않음. 변수에 세대를 기록하여 이를 해결
from step14 import *


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):  # 입력받은 데이터가 ndarray가 아니면 에러 발생
                raise TypeError("{}는 지원하지 않습니다.".format(type(data)))

        self.data = data
        self.grad = None  # data와 대응하는 미분값
        self.creator = None
        self.generation = 0  # 세대 수 기록

    def set_creator(self, func):  # 변수의 관점에서 함수는 자신을 만들어낸 창조자(creator)
        self.creator = func
        self.generation = func.generation + 1  # 부모 함수

    def backward(self):  # 반복문을 사용해 구현.
        if self.grad is None:  # 기울기가 없을때 자동으로 초기 미분값 생성
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()  # 같은 함수가 여러번 추가되는것을 방지하기 위한 set

        def add_func(f):  # 사용된 함수 리스트를 세대 순으로 정렬
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)  #

        add_func(self.creator)

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
                    add_func(x.creator)

    def cleargrad(self):  # 같은 변수를 다른 계산에 사용할 때 문제가 생기기 때문에 기울기 초기화 필요
        self.grad = None


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]  # 입력을 가변 길이 인수로 변경
        ys = self.forward(*xs)  # 애스터리스크로 언팩
        if not isinstance(ys, tuple):  # 튜플이 아니면 튜플로 묶어줌
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])  # 입력 변수가 둘 이상일 때 가장 큰 generation을 가져옴
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


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(x.grad)