#!/usr/bin/env python3
import numpy as np

n = 24

# #(arguments) <= InsBlockSize - 3
InsBlockSize = 9

# m is a multiple of InsBlockSize
m = InsBlockSize * (n * n // InsBlockSize)

# State[k] in {-M, ..., M}
M = 100000

# learner's variable internal state
State = [0] * m

InsSet = [i for i in range(0, n)]

# brains
Brain = [m * [[(1 / n)] * n]] * 2

# InstructionPointer
InsPointer = 0

Stack = [[]] * 2
# stack elements look like that:
# [t, RL(t), (c1, Left(c1)), ...]
# where t - checkpoint time
#       RL(t) - reward until time t
#       c1 - idx of modified column
#       Left(c1) - previous Left-column

BlockSSA = [False] * 2

# SSA Calls
# https://people.idsia.ch/~juergen/mljssalevin/node2.html
def SSA(left):
  if not BlockSSA[left]:
    BlockSSA[left] = True
    while True:

      # t is a new checkpoint
      t = currTime

      # backtracking

      # trivial case
      if len(Stack[left]) == 0:
        Stack[left].append([t, RL(t)])
        break
      else:
        # t' and t''
        t1, RL(t1) = Stack[left][-1][0], Stack[left][-1][1]
        if len(StackLeft) >= 2:
          t2, RL(t2) = Stack[left][-2][0], Stack[left][-2][1]
        else:
          t2, RL_t2 = 0, 0

        # main induct rule
        if (RL(t) - RL(t1)) / (t - t1) > (RL(t) - RL(t2)) / (t - t2):
          Stack[left].append([t, RL(t)])
          break
        else:
          # pop t1 block, restore everything as it was before t1
          lblock = Stack[left].pop()
          modifs = lblock[2]
          for modif in modifs:
            # restore modif
            Brain[modif[0]] = modif[1]



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

class InsExec():
  def ins_Jmpl(self, params, clean_params):
    assert len(clean_params) == 3
    if State[clean_params[0]] < State[clean_params[1]]:
      InsPointer = clean_params[2] - (clean_params[2] % InsBlockSize)
  def ins_Jmpeq(self, params, clean_params):
    assert len(clean_params) == 3
    if State[clean_params[0]] == State[clean_params[1]]:
      InsPointer = clean_params[2] - (clean_params[2] % InsBlockSize)
  def ins_Add(self, params, clean_params):
    assert len(clean_params) == 3
    State[clean_params[2]] = (State[clean_params[0]] + State[clean_params[1]]) % M
  def ins_Sub(self, params, clean_params):
    assert len(clean_params) == 3
    State[clean_params[2]] = (State[clean_params[0]] - State[clean_params[1]]) % M
  def ins_Mul(self, params, clean_params):
    assert len(clean_params) == 3
    State[clean_params[2]] = (State[clean_params[0]] * State[clean_params[1]]) % M
  def ins_Div(self, params, clean_params):
    assert len(clean_params) == 3
    if State[clean_params[1]] != 0:
      State[clean_params[2]] = (State[clean_params[0]] // State[clean_params[1]]) % M
  def ins_Mov(self, params, clean_params):
    assert len(clean_params) == 2
    State[clean_params[1]] = State[clean_params[0]]
  def ins_Init(self, params, clean_params):
    assert len(clean_params) == 2
    State[clean_params[1]] = clean_params[0]

  def exec_ins(self, ins_idx, params):
    ins_name = 'ins_' + tab_ins[ins_idx][0]
    assert tab_ins[ins_idx][1] == len(params)
    ins_method = getattr(self, ins_name)
    clean_params = []
    for i in range(0, len(params), 2):
      assert i + 1 < len(params) 
      clean_params.append((params[i] * n + params[i+1]) % m)
    ins_method(params, clean_params)
    
    
insExec = InsExec()
while True:
  # select instruction head a[j] with max? probability Q(IP, j)
  ins_idx = getDecision(InsPointer)

  # select arguments 
  num_params = tab_ins[ins_idx][1]
  params = []
  for i in range(1, num_params + 1):
    param_idx = getDecision(InsPointer + i)
    params.append(param_idx)
  # execute the instruction
  insExec.exec_ins(ins_idx, params)

  # external reward?
  # inputs?
  
  if tab_ins[ins_idx][0] != "Jmpl" and tab_ins[ins_idx][0] != "Jmpeq":
    w1 = getDecision(InsPointer + 7)
    w2 = getDecision(InsPointer + 8)
    w = (w1 * n + w2) % m
    InsPointer = w - (w % InsBlockSize)
