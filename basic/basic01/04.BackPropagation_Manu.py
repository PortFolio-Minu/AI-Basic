import numpy as np

class Variable:
  def __init__(self, data):
    self.data = data
    self.grad = None

class Function:
  def __call__(self, input):
    x = input.data # 데이터를 꺼낸다
    y = x**2 # 실제 계산
    output = Variable(y)
    self.input = input
    return output

  def forward(self, x):
    raise NotImplementedError()

  def backward(self, gy):
    raise NotImplementedError()

def numerical_diff(f, x, eps=1e-4):
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data - y0.data) / (2* eps)

class Square(Function):
  def forward(self, x):
    y = x**2
    return y 
  
  def backward(self, gy):
    x = self.input.data
    gx = 2*x*gy
    return gx

class Exp(Function):
  def forward(self, x):
    y = np.exp(x)
    return y
  
  def backward(self, gy):
      x = self.input.data
      gx = np.exp(x) * gy
      return gx

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)


y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)