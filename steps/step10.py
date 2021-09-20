import unittest
from step04 import *
from step09 import *


class SquareTest(unittest.TestCase):
    def test_forward(self):  # 이름이 test로 시작해야 함
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)  # y값과 기대값이 같은지 판정

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    # 수치 미분과 역전파로 구한 값을 비교
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


if __name__ == "__main__":
    unittest.main()
