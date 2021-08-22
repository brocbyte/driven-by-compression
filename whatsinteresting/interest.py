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

# environ reward
ExternalReward = 0

# curiosity reward
from collections import defaultdict
InternalReward = [defaultdict(lambda: 0)] * 2

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
def SSA(right):
  if not BlockSSA[right]:
    BlockSSA[right] = True
    while True:

      # t is a new checkpoint
      t = currTime

      # backtracking

      # trivial case
      if len(Stack[right]) == 0:
        Stack[right].append([t, InternalReward[right][t]])
        break
      else:
        # t' and t''
        t1, InternalReward[right][t1] = Stack[right][-1][0], Stack[right][-1][1]
        if len(StackLeft) >= 2:
          t2, InternalReward[right][t2] = Stack[right][-2][0], Stack[right][-2][1]
        else:
          t2, InternalReward[right][t2] = 0, 0

        # main induct rule
        if (InternalReward[right][t] - InternalReward[right][t1]) / (t - t1) > (InternalReward[right][t] - InternalReward[right][t2]) / (t - t2):
          Stack[right].append([t, InternalReward[right][t]])
          break
        else:
          # pop t1 block, restore everything as it was before t1
          lblock = Stack[right].pop()
          modifs = lblock[2]
          for modif in modifs:
            # restore modif
            Brain[modif[0]] = modif[1]



# collective decision function
def f(x, y):
  return x * y

def Q(i, j):
  return f(Brain[1][i][j], Brain[0][i][j]) / sum(f(x, y) for (x, y) in zip(Brain[1][i], Brain[0][i]))

def getDecision(i):
  return np.argmax([Q(i, j) for j in range(0, n)])

tab_ins = [
  # instructions operating on internal state only
  ["Jmpl", 6], ["Jmpeq", 6],
  ["Add", 6], ["Sub", 6], ["Mul", 6], ["Div", 6],
  ["Mov", 4], ["Init", 4],
  # bet!
  ["Bet", 4],
  # introspective instructions
  ["GetLeft", 3], ["GetRight", 3],
  # SSA-enabling
  ["EnableSSALeft", 1], ["EnableSSARight", 1],
  # primitive learning algorithms
  ["IncProbLeft", 3],
  # interaction with the environment
  ["MoveAgent", 0], ["SetDirection", 1]
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

  def ins_Bet(self, params, clean_params):
    assert len(params) == 6
    # c = d
    if params[4] == params[5]:
      return
    if State[clean_params[1]] == State[clean_params[0]]:
      # give reward c to Left and -c to Right
      InternalReward[0] += c
      InternalReward[1] -= c
    else:
      # give reward -c to Left and c to Right
      InternalReward[0] -= c
      InternalReward[1] += c
    # surprise rewards become visible in the form of inputs
    State[7] = c
  def ins_GetLeft(self, params, clean_params):
    assert len(params) == 3 and len(clean_params) == 1
    State[clean_params[0]] = math.round(M * Brain[0][clean_params[0]][params[2]])
  def ins_GetRight(self, params, clean_params):
    assert len(params) == 3 and len(clean_params) == 1
    State[clean_params[0]] = math.round(M * Brain[1][clean_params[0]][params[2]])
  def ins_EnableSSALeft(self, params, clean_params):
    assert len(params) == 1 and len(clean_params) == 0
    if params[0] < 10:
      BlockSSALeft = False
  def ins_EnableSSARight(self, params, clean_params):
    assert len(params) == 1 and len(clean_params) == 0
    if params[0] < 10:
      BlockSSARight = False

  lambda_const = 0.3
  MinProb = 0.004
  def saveBrainCol(self, right, column):
    if Stack[right][-1][2].get(column, default = None) != None:
      Stack[right][-1][2][column] = Brain[0][column].copy()
    

  def ins_IncPropLeft(self, params, clean_params):
    SSA(right = False)
    assert len(params) == 3 and len(clean_params) == 1 and len(Stack[0]) > 0
    saveBrainCol(right = False, column = clean_params[0])

    for k in range(0, len(Brain[0][clean_params[0]])):
      if k == params[2]:
        Brain[0][clean_params[0]][k] = 1 - lambda_const * (1 - Brain[0][clean_params[0]][k])
      else:
        Brain[0][clean_params[0]][k] *= lambda_const

    # we've saved it already
    if any(elem < MinProb for elem in Brain[0][clean_params[0]]):
      Brain[0][clean_params[0]] = Stack[0][-1][2][clean_params[0]]

  def ccw(A,B,C):
      return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
  def intersect(AB, CD):
      [A, B] = AB
      [C, D] = CD
      return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
  def updateInputs():
    x, y, d = State[0], State[1], State[2] / 100 * (2 * math.pi)
    front_v = [[x, y], [24 * (x + cos(d)), 24 * (y + sin(d))]]
    State[3] = 24 if any(intersect(wall, front_v) for wall in walls) else 0

    right_v  = [[x, y], [24 * (x + cos(d - math.pi / 2)), 24 * (y + sin(d - math.pi / 2))]]
    State[4] = 24 if any(intersect(wall, right_v) for wall in walls) else 0

    back_v = [[x, y], [-24 * (x + cos(d)), -24 * (y + sin(d))]]
    State[5] = 24 if any(intersect(wall, back_v) for wall in walls) else 0

    left_v  = [[x, y], [24 * (x + cos(d + math.pi / 2)), 24 * (y + sin(d + math.pi / 2))]]
    State[6] = 24 if any(intersect(wall, left_v) for wall in walls) else 0



  from math import sin, cos
  import math
  def ins_MoveAgent(self, params, clean_params):
    Vel = 12
    assert len(params) == 0 and len(clean_params) == 0
    x, y, d = State[0], State[1], State[2] / 100 * (2 * math.pi)
    move = [[x, y], [Vel * (x + cos(d)), Vel * (y + sin(d))]]
    if not any(intersect(wall, move) for wall in walls):
      State[0], State[1] = int(move[1][0]), int(move[1][1])
    updateInputs()
  def ins_SetDirection(self, params, clean_params):
    State[2] = params[0] / n * 100
    updateInputs()

          

  def exec_ins(self, ins_idx, params):
    ins_name = 'ins_' + tab_ins[ins_idx][0]
    assert tab_ins[ins_idx][1] == len(params)
    ins_method = getattr(self, ins_name)
    clean_params = []
    for i in range(0, len(params), 2):
      assert i + 1 < len(params) 
      clean_params.append((params[i] * n + params[i+1]) % m)
    ins_method(params, clean_params)
    


import tkinter as tk
root = tk.Tk()
delta_t = 20
size = 1000
canvas = tk.Canvas(root, width = size, height = size)
canvas.pack()

x, y = 0, 0
dot = canvas.create_oval(x,y,x+20,y+20,outline="white",fill="green") 

def redraw():
  canvas.after(delta_t,redraw)
  canvas.move(dot, dx, dy)
  canvas.update()

canvas.after(delta_t, redraw)
root.mainloop()




    
insExec = InsExec()
while True:
  # select instruction head a[j] with max? probability Q(IP, j)
  ins_idx = getDecision(InsPointer)

  # select arguments 
  num_params = tab_ins[ins_idx][1]
  params = []

  for i in range(1, num_params + 1):
    param = getDecision(InsPointer + i)
    params.append(param)

  if tab_ins[ins_idx][0] == "Bet":
    c = np.argmax([(Brain[0][InsPointer + 5][j] / sum(Brain[0][InsPointer + 5])) for j in range(0, n)])
    c = 1 if c > (n / 2) else -1
    params.append(c)

    d = np.argmax([(Brain[1][InsPointer + 5][j] / sum(Brain[1][InsPointer + 5])) for j in range(0, n)])
    d = 1 if d > (n / 2) else -1
    params.append(d)

   

  # execute the instruction
  insExec.exec_ins(ins_idx, params)

  # external reward
  if ExternalReward != 0:
    State[8] = ExternalReward

  # if an input has changed S0-S8, then shift S0-S80 to S9-S89
  
  if tab_ins[ins_idx][0] != "Jmpl" and tab_ins[ins_idx][0] != "Jmpeq":
    w1 = getDecision(InsPointer + 7)
    w2 = getDecision(InsPointer + 8)
    w = (w1 * n + w2) % m
    InsPointer = w - (w % InsBlockSize)
