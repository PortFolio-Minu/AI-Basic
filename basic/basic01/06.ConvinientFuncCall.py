import numpy as np


## Classes

class Variable:
  def __init__(self, data):
    if data is not None:
      if not isinstance(data, np.ndarray):
        raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
    self.data = data
    self.grad = None
    self.creator = None

  
  
  def set_creator(self, func):
    self.creator = func

  def backward(self):
    if self.grad is None:
      self.grad = np.ones_like(self.data)
    
    funcs = [self.creator]

    while funcs :
      f = funcs.pop()
      x, y = f.input, f.output
      x.grad = f.backward(y.grad)

      if x.creator is not None:
        funcs.append(x.creator)


class Function:
  def __call__(self, input):
    x = input.data # 데이터를 꺼낸다
    y = x**2 # 실제 계산
    output = Variable(as_array(y))
    output.set_creator(self)  # 출력 변수에 참조자를 설정
    self.input = input
    self.output = output
    return output

  def forward(self, x):
    raise NotImplementedError()

  def backward(self, gy):
    raise NotImplementedError()

class Square(Function):
  def forward(self, x):
    y = x**2
    return y 
  
  def backward(self, gy):
    x = self.input.data
    gx = 2* x * gy
    return gx

class Exp(Function):
  def forward(self, x):
    y = np.exp(x)
    return y
  
  def backward(self, gy):
    x = self.input.data
    gx = np.exp(x) * gy
    return gx


## Functions

def numerical_diff(f, x, eps=1e-4):
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data - y0.data) / (2* eps)

def square(x):
  return Square()(x)

def exp(x):
  return Exp()(x)

def as_array(x):
    if np.isscalar(x):
      return np.array(x)
    return x

x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)

