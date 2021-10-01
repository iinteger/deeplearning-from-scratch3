# 변수 이름 추가
# variable 클래스를 numpy array 쓰듯이 사용할 수 있도록 property 추가
from step18 import *


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):  # 입력받은 데이터가 ndarray가 아니면 에러 발생
                raise TypeError("{}는 지원하지 않습니다.".format(type(data)))

        self.data = data
        self.name = name
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

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n"+' '*9)
        return "variable(" + p + ")"

if __name__ == "__main__":
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    print(len(x))
    print(x)