class Variable:
    def __init__(self, data):
        self.data = data


import numpy as np

# data = np.array(1.0)
# x = Variable(data)  # variable은 데이터를 담는 컨테이너. 진짜 데이터는 field인 data에 있음
# print(x.data)