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

class Square(Function):
  def forward(self, x):
    return x**2

class Exp(Function):
  def forward(self, x):
    return np.exp(x)
  
x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(y))
print(y.data)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(y.data)

