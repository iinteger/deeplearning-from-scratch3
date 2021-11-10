# error 있음 100.0이 나와야 하는데 4.0이 나옴

import numpy as np
from dezero import Variable


x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)