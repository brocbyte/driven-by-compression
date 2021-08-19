#!/usr/bin/env python3
import numpy as np

n = 24

# #(arguments) <= InsBlockSize - 3
InsBlockSize = 2

# m is a multiple of InsBlockSize
m = InsBlockSize * (n * n // InsBlockSize)

# State[k] in {-M, ..., M}
M = 100000

# learner's variable internal state
State = [0] * m

InsSet = [i for i in range(0, n)]

# brains
RightBr = m * [[(1 / n)] * n]
LeftBr = m * [[(1 / n)] * n]

# InstructionPointer
InsPointer = 0

StackRight = []
StackLeft = []

BlockSSALeft = False
BlockSSARight = False

# collective decision function
def f(x, y):
  return x * y

def Q(i, j):
  return f(RightBr[i][j], LeftBr[i][j]) / sum(f(x, y) for (x, y) in zip(RightBr[i], LeftBr[i]))

def getDecision(i):
  return np.argmax([Q(i, j) for j in range(0, n)])

tab_ins = [
  # instructions operating on internal state only
  ["Jmpl", 6], ["Jmpeq", 6],
  ["Add", 6], ["Sub", 6], ["Mul", 6], ["Div", 6],
  ["Mov", 4], ["Init", 4]
]

class Ins():
  def ins_Jmpl(self, params, clean_params):
    # yep...
  def exec_ins(self, ins_idx, params):
    ins_name = 'ins_' + tab_ins[ins_idx][0]
    assert tab_ins[ins_idx][1] == len(params)
    ins_method = getattr(self, ins_name)
    clean_params = []
    for i in range(0, len(params), 2):
      assert i + 1 < len(params) 
      clean_params.append((params[i] * n + params[i+1]) % m)
    ins_method(params, clean_params)
    
    




while True:
  # select instruction head a[j] with max? probability Q(IP, j)
  ins_idx = getDecision(InsPointer)

  # select arguments 
  num_params = tab_ins[ins_idx][1]
  for i in range(1, num_params + 1):
    param_idx = getDecision(InsPointer + i)
    params.append(param_idx)
  # execute the instruction
  exec_ins(ins_idx, params)

