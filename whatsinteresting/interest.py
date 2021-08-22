#!/usr/bin/env python3
import numpy as np
from collections import defaultdict

class Agent():
  def __init__(self):
    n = 24
    InsBlockSize = 9
    m = InsBlockSize * (n * n // InsBlockSize)
    M = 100000

    self.State = [0] * m
    self.Brain = [m * [[(1 / n)] * n]] * 2
    self.ExternalReward = 0
    self.InternalReward = [defaultdict(lambda: 0)] * 2
    self.InsPointer = 0
    self.Stack = [[]] * 2
    # stack elements look like that:
    # [t, RL(t), (c1, Left(c1)), ...]
    # where t - checkpoint time
    #       RL(t) - reward until time t
    #       c1 - idx of modified column
    #       Left(c1) - previous Left-column

    self.BlockSSA = [False] * 2

  # SSA Calls
  # https://people.idsia.ch/~juergen/mljssalevin/node2.html
  def SSA(right):
    if not self.BlockSSA[right]:
      self.BlockSSA[right] = True
      while True:

        # t is a new checkpoint
        t = currTime

        # backtracking

        # trivial case
        if len(self.Stack[right]) == 0:
          self.Stack[right].append([t, self.InternalReward[right][t]])
          break
        else:
          # t' and t''
          t1, self.InternalReward[right][t1] = self.Stack[right][-1][0], self.Stack[right][-1][1]
          if len(self.Stack[right]) >= 2:
            t2, self.InternalReward[right][t2] = self.Stack[right][-2][0], self.Stack[right][-2][1]
          else:
            t2, self.InternalReward[right][t2] = 0, 0

          # main induct rule
          if (self.InternalReward[right][t] - self.InternalReward[right][t1]) / (t - t1) > (self.InternalReward[right][t] - self.InternalReward[right][t2]) / (t - t2):
            self.Stack[right].append([t, self.InternalReward[right][t]])
            break
          else:
            # pop t1 block, restore everything as it was before t1
            lblock = self.Stack[right].pop()
            modifs = lblock[2]
            for modif in modifs:
              # restore modif
              self.Brain[modif[0]] = modif[1]

  def Q(i, j):
    def f(x, y):
      return x * y
    return f(self.Brain[1][i][j], self.Brain[0][i][j]) / sum(f(x, y) for (x, y) in zip(self.Brain[1][i], self.Brain[0][i]))

  def getDecision(i):
    return np.argmax([Q(i, j) for j in range(0, n)])

  def ins_Jmpl(self, params, clean_params):
    assert len(clean_params) == 3
    if self.State[clean_params[0]] < self.State[clean_params[1]]:
      self.InsPointer = clean_params[2] - (clean_params[2] % InsBlockSize)
  def ins_Jmpeq(self, params, clean_params):
    assert len(clean_params) == 3
    if self.State[clean_params[0]] == self.State[clean_params[1]]:
      self.InsPointer = clean_params[2] - (clean_params[2] % InsBlockSize)
  def ins_Add(self, params, clean_params):
    assert len(clean_params) == 3
    self.State[clean_params[2]] = (self.State[clean_params[0]] + self.State[clean_params[1]]) % M
  def ins_Sub(self, params, clean_params):
    assert len(clean_params) == 3
    self.State[clean_params[2]] = (self.State[clean_params[0]] - self.State[clean_params[1]]) % M
  def ins_Mul(self, params, clean_params):
    assert len(clean_params) == 3
    self.State[clean_params[2]] = (self.State[clean_params[0]] * self.State[clean_params[1]]) % M
  def ins_Div(self, params, clean_params):
    assert len(clean_params) == 3
    if self.State[clean_params[1]] != 0:
      self.State[clean_params[2]] = (self.State[clean_params[0]] // self.State[clean_params[1]]) % M
  def ins_Mov(self, params, clean_params):
    assert len(clean_params) == 2
    self.State[clean_params[1]] = self.State[clean_params[0]]
  def ins_Init(self, params, clean_params):
    assert len(clean_params) == 2
    self.State[clean_params[1]] = clean_params[0]

  def ins_Bet(self, params, clean_params):
    assert len(params) == 6
    c, d = params[4], params[5]
    if c == d:
      return
    if self.State[clean_params[1]] == self.State[clean_params[0]]:
      # give reward c to Left and -c to Right
      self.InternalReward[0] += c
      self.InternalReward[1] -= c
    else:
      # give reward -c to Left and c to Right
      self.InternalReward[0] -= c
      self.InternalReward[1] += c
    # surprise rewards become visible in the form of inputs
    self.State[7] = c
  def ins_GetLeft(self, params, clean_params):
    assert len(params) == 3 and len(clean_params) == 1
    self.State[clean_params[0]] = math.round(M * self.Brain[0][clean_params[0]][params[2]])
  def ins_GetRight(self, params, clean_params):
    assert len(params) == 3 and len(clean_params) == 1
    self.State[clean_params[0]] = math.round(M * self.Brain[1][clean_params[0]][params[2]])
  def ins_EnableSSALeft(self, params, clean_params):
    assert len(params) == 1 and len(clean_params) == 0
    if params[0] < 10:
      self.BlockSSA[0] = False
  def ins_EnableSSARight(self, params, clean_params):
    assert len(params) == 1 and len(clean_params) == 0
    if params[0] < 10:
      self.BlockSSA[1] = False

  lambda_const = 0.3
  MinProb = 0.004
  def saveBrainCol(self, right, column):
    if self.Stack[right][-1][2].get(column, default = None) != None:
      self.Stack[right][-1][2][column] = self.Brain[0][column].copy()
    

  def ins_IncPropLeft(self, params, clean_params):
    SSA(right = False)
    assert len(params) == 3 and len(clean_params) == 1 and len(self.Stack[0]) > 0
    saveBrainCol(right = False, column = clean_params[0])

    for k in range(0, len(self.Brain[0][clean_params[0]])):
      if k == params[2]:
        self.Brain[0][clean_params[0]][k] = 1 - lambda_const * (1 - self.Brain[0][clean_params[0]][k])
      else:
        self.Brain[0][clean_params[0]][k] *= lambda_const

    # we've saved it already
    if any(elem < MinProb for elem in self.Brain[0][clean_params[0]]):
      self.Brain[0][clean_params[0]] = self.Stack[0][-1][2][clean_params[0]]

  def ccw(A,B,C):
      return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
  def intersect(AB, CD):
      [A, B] = AB
      [C, D] = CD
      return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

  from math import sin, cos, pi
  def updateInputs():
    x, y, d = self.State[0], self.State[1], self.State[2] / 100 * (2 * pi)
    front_v = [[x, y], [24 * (x + cos(d)), 24 * (y + sin(d))]]
    self.State[3] = 24 if any(intersect(wall, front_v) for wall in walls) else 0

    right_v  = [[x, y], [24 * (x + cos(d - pi / 2)), 24 * (y + sin(d - pi / 2))]]
    self.State[4] = 24 if any(intersect(wall, right_v) for wall in walls) else 0

    back_v = [[x, y], [-24 * (x + cos(d)), -24 * (y + sin(d))]]
    self.State[5] = 24 if any(intersect(wall, back_v) for wall in walls) else 0

    left_v  = [[x, y], [24 * (x + cos(d + math.pi / 2)), 24 * (y + sin(d + math.pi / 2))]]
    self.State[6] = 24 if any(intersect(wall, left_v) for wall in walls) else 0



  def ins_MoveAgent(self, params, clean_params):
    Vel = 12
    assert len(params) == 0 and len(clean_params) == 0
    x, y, d = self.State[0], self.State[1], self.State[2] / 100 * (2 * pi)
    move = [[x, y], [Vel * (x + cos(d)), Vel * (y + sin(d))]]
    if not any(intersect(wall, move) for wall in walls):
      self.State[0], self.State[1] = int(move[1][0]), int(move[1][1])
    updateInputs()
  def ins_SetDirection(self, params, clean_params):
    self.State[2] = params[0] / n * 100
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

  def act(self):
    tab_ins = [
        ["Jmpl", 6], ["Jmpeq", 6],
        ["Add", 6], ["Sub", 6], ["Mul", 6], ["Div", 6],
        ["Mov", 4], ["Init", 4],
        ["Bet", 4],
        ["GetLeft", 3], ["GetRight", 3],
        ["EnableSSALeft", 1], ["EnableSSARight", 1],
        ["IncProbLeft", 3],
        ["MoveAgent", 0], ["SetDirection", 1]
    ]

    # select instruction head a[j] with max? probability Q(IP, j)
    ins_idx = getDecision(self.InsPointer)

    # select arguments 
    params = []
    for i in range(1, tab_ins[ins_idx][1] + 1):
      param = getDecision(self.InsPointer + i)
      params.append(param)

    # take care of Bet!
    if tab_ins[ins_idx][0] == "Bet":
      c = np.argmax([(self.Brain[0][self.InsPointer + 5][j] / sum(self.Brain[0][self.InsPointer + 5])) for j in range(0, n)])
      c = 1 if c > (n / 2) else -1
      params.append(c)

      d = np.argmax([(self.Brain[1][self.InsPointer + 5][j] / sum(self.Brain[1][self.InsPointer + 5])) for j in range(0, n)])
      d = 1 if d > (n / 2) else -1
      params.append(d)

     
    exec_ins(ins_idx, params)

    # external reward
    if self.ExternalReward != 0:
      self.State[8] = self.ExternalReward

    # TODO if an input has changed S0-S8, then shift S0-S80 to S9-S89
    
    if tab_ins[ins_idx][0] != "Jmpl" and tab_ins[ins_idx][0] != "Jmpeq":
      w1 = getDecision(self.InsPointer + 7)
      w2 = getDecision(self.InsPointer + 8)
      w = (w1 * n + w2) % m
      self.InsPointer = w - (w % InsBlockSize)
