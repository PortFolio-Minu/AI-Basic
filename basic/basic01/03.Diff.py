import numpy as np

class Variable:
  def __init__(self, data):
    self.data = data

class Function:
  def __call__(self, input):
    x = input.data # 데이터를 꺼낸다
    y = x**2 # 실제 계산
    output = Variable(y)
    return output

  def forward(self, x):
    raise NotImplementedError()

def numerical_diff(f, x, eps=1e-4):
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data - y0.data) / (2* eps)

class Square(Function):
  def forward(self, x):
    return x**2

class Exp(Function):
  def forward(self, x):
    return np.exp(x)


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)

print(dy)

