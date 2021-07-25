import numpy as np

class Variable:
  def __init__(self, data):
    self.data = data
    self.grad = None
    self.creator = None
  
  def set_creator(self, func):
    self.creator = func

  def backward(self):
    f = self.creator # 1. 함수를 가져온다
    if f is not None:
      x = f.input # 2. 함수의 입력을 가져온다
      x.grad = f.backward(self.grad) # 함수의 backward 메서드를 호출
      x.backward() # 재귀 : 하나의 앞 변수의 backward 메서드를 호출


class Function:
  def __call__(self, input):
    x = input.data # 데이터를 꺼낸다
    y = x**2 # 실제 계산
    output = Variable(y)
    output.set_creator(self)  # 출력 변수에 참조자를 설정
    self.input = input
    self.output = output
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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 계산 그래프의 노드들을 꺼꾸로 거술러 올라간다
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x


## 1. 수동방식
'''
y.grad = np.array(1.0)
C = y.creator # 1. 함수를 가져온다
b = C.input # 2. 함수의 입력을 가져온다
b.grad = C.backward(y.grad) # 3. 함수의 backward 메서드를 호출

B = b.creator # 1. 함수를 가져온다
a = B.input # 2. 함수의 입력을 가져온다
a.grad = B.backward(b.grad) # 3. 함수의 backward 메서드를 호출한다

A = a.creator # 1. 함수를 가져온다
x = A.input # 2. 함수의 입력을 가져온다.
x.grad = A.backward(a.grad) # 3. 함수의 backward 메서드를 호출한ㄷ
print(x.grad)
'''

## 2.재귀호출방식

y.grad = np.array(1.0)
y.backward()
print(x.grad)