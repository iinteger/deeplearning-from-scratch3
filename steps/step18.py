# 보통 중간 변수의 미분값은 필요하지 않으므로 중간 변수들의 미분값을 선택적으로 저장하도록 변경
# 역전파가 필요 없는 경우의 모드 추가 ex) 추론만 할 때
import numpy as np

from step17 import *
import contextlib


# 설정 데이터는 한군데에만 존재하는게 좋음
# 인스턴스는 여러개 생성이 가능하지만 클래스는 하나만 존재할 수 있기 때문에 인스턴스로 만들지 않고 클래스 상태로 이용
class Config:
    enable_backprop = True  # 역전파가 가능할지에 대한 여부. True일 때 역전파 활성 모드


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield  # 이 전에는 전처리 로직, 이 뒤로는 후처리 로직
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


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

    def backward(self, retain_grad=False):  # 반복문을 사용해 구현.
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
            gys = [output().grad for output in f.outputs]  # 순환 참조 해결
            gxs = f.backward(*gys)  # 역전파
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):  # 역전파로 흘려보내는 미분값을 Variable의 grad 필드에 담음
                if x.grad is None:  # 미분값을 처음 설정할 때는 출력에서 전달되는 미분값을 그대로 사용해야함
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:  # True일 때 모든 변수가 미분 결과를 유지함
                for y in f.outputs:
                    y().grad = None  # y는 약한 참조

    def cleargrad(self):  # 같은 변수를 다른 계산에 사용할 때 문제가 생기기 때문에 기울기 초기화 필요
        self.grad = None


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]  # 입력을 가변 길이 인수로 변경
        ys = self.forward(*xs)  # 애스터리스크로 언팩
        if not isinstance(ys, tuple):  # 튜플이 아니면 튜플로 묶어줌
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # 역전파가 활성화됐을때만
            self.generation = max([x.generation for x in inputs])  # 입력 변수가 둘 이상일 때 가장 큰 generation을 가져옴
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

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
    with using_config("enable_backprop", False):
        x = Variable(np.array(2.0))
        y = square(x)
