#!/usr/bin/env python3
import numpy as np
import math
from collections import defaultdict
from intersect import intersect
import random

class Agent():
  n = 24
  InsBlockSize = 9
  m = InsBlockSize * (n * n // InsBlockSize)
  M = 100000
  def __init__(self, walls = []):
    self.State = [0] * Agent.m
    self.State[0], self.State[1] = 500, 500
    self.Brain = []
    for i in range(0, 2):
      b = []
      for j in range(0, Agent.m):
        c = []
        for k in range(0, Agent.n):
          c.append(1 / Agent.n)
        b.append(c)
      self.Brain.append(b)

    self.ER = 0
    # we should maintain RL(t), RR(t) - current time, and RLs, RRs on Stacks
    self.IR = [0] * 2
    self.InsPtr = 0
    self.Stack = [[], []]
    self.walls = walls
    self.time = 1
    # stack elements look like that:
    # [t, RL(t), (c1, Left(c1)), ...]
    # where t - checkpoint time
    #       RL(t) - reward until time t
    #       c1 - idx of modified column
    #       Left(c1) - previous Left-column

    self.BlockSSA = [False] * 2
    self.InsStat = defaultdict(int)

  # SSA Calls
  # https://people.idsia.ch/~juergen/mljssalevin/node2.html
  def SSA(self, right):
    if not self.BlockSSA[right]:
      self.BlockSSA[right] = True
      while True:

        # t is a new checkpoint
        t = self.time

        # backtracking

        # trivial case
        if len(self.Stack[right]) == 0:
          self.Stack[right].append([t, self.IR[right], dict()])
          break
        else:
          # t' and t''
          t1, IRt1 = self.Stack[right][-1][0], self.Stack[right][-1][1]
          if len(self.Stack[right]) >= 2:
            t2, IRt2 = self.Stack[right][-2][0], self.Stack[right][-2][1]
          else:
            t2, IRt2 = 0, 0

          # main induct rule
          assert t != t2
          assert t != t1
          if (self.IR[right] - IRt1) / (t - t1) > (self.IR[right] - IRt2) / (t - t2):
            self.Stack[right].append([t, self.IR[right], dict()])
            break
          else:
            # pop t1 block, restore everything as it was before t1
            lblock = self.Stack[right].pop()
            modifs = lblock[2]
            for k, v in modifs.items():
              # restore modif
              assert len(self.Brain[right][k]) == len(v)
              for i in range(0, len(self.Brain[right][k])):
                self.Brain[right][k][i] = v[i]
      self.time += 1
      assert len(self.Stack[right]) > 0

  def ins_Jmpl(self, params, clean_params):
    if self.State[clean_params[0]] < self.State[clean_params[1]]:
      self.InsPtr = clean_params[2] - (clean_params[2] % Agent.InsBlockSize)
  def ins_Jmpeq(self, params, clean_params):
    if self.State[clean_params[0]] == self.State[clean_params[1]]:
      self.InsPtr = clean_params[2] - (clean_params[2] % Agent.InsBlockSize)

  def ins_Add(self, params, clean_params):
    if clean_params[2] > 8:
      self.State[clean_params[2]] = (self.State[clean_params[0]] + self.State[clean_params[1]]) % Agent.M
  def ins_Sub(self, params, clean_params):
    if clean_params[2] > 8:
      self.State[clean_params[2]] = (self.State[clean_params[0]] - self.State[clean_params[1]]) % Agent.M
  def ins_Mul(self, params, clean_params):
    if clean_params[2] > 8:
      self.State[clean_params[2]] = (self.State[clean_params[0]] * self.State[clean_params[1]]) % Agent.M
  def ins_Div(self, params, clean_params):
    if clean_params[2] > 8:
      if self.State[clean_params[1]] != 0:
        self.State[clean_params[2]] = (self.State[clean_params[0]] // self.State[clean_params[1]]) % Agent.M

  def ins_Mov(self, params, clean_params):
    if clean_params[1] > 8:
      self.State[clean_params[1]] = self.State[clean_params[0]]
  def ins_Init(self, params, clean_params):
    if clean_params[1] > 8:
      self.State[clean_params[1]] = clean_params[0]

  def ins_Bet(self, params, clean_params):
    c, d = params[4], params[5]
    if c == d:
      return
  
    if self.State[clean_params[1]] == self.State[clean_params[0]]:
      self.IR[0] += c
      self.IR[1] -= c
    else:
      self.IR[0] -= c
      self.IR[1] += c
    # surprise rewards become visible in the form of inputs
    self.State[7] = c

  def ins_GetLeft(self, params, clean_params):
    if clean_params[0] > 8:
      self.State[clean_params[0]] = math.ceil(Agent.M * self.Brain[0][clean_params[0]][params[2]])
  def ins_GetRight(self, params, clean_params):
    if clean_params[0] > 8:
      self.State[clean_params[0]] = math.ceil(Agent.M * self.Brain[1][clean_params[0]][params[2]])

  def ins_EnableSSALeft(self, params, clean_params):
    if params[0] < 10:
      self.BlockSSA[0] = False
  def ins_EnableSSARight(self, params, clean_params):
    if params[0] < 10:
      self.BlockSSA[1] = False

  lambda_const = 0.3
  MinProb = 0.004
  def saveBrainCol(self, right, column):
    assert len(self.Stack[right]) > 0
    if column not in self.Stack[right][-1][2]:
      self.Stack[right][-1][2][column] = self.Brain[right][column].copy()
    
  def IncProb(self, right, params, clean_params):
    self.SSA(right)
    x = clean_params[0]
    self.saveBrainCol(right, x)
    for k in range(0, len(self.Brain[right][x])):
      if k == params[2]:
        self.Brain[right][x][k] = 1 - Agent.lambda_const * (1 - self.Brain[right][x][k])
      else:
        self.Brain[right][x][k] *= Agent.lambda_const

    # we've saved it already
    if any(elem < Agent.MinProb for elem in self.Brain[right][x]):
      assert len(self.Brain[right][x]) == len(self.Stack[right][-1][2][x])
      for k in range(0, len(self.Brain[right][x])):
        self.Brain[right][x][k] = self.Stack[right][-1][2][x][k]
    assert all(all(elem >= Agent.MinProb for elem in self.Brain[right][x]) for x in range(0, Agent.m))

  def DecProb(self, right, params, clean_params):
    self.SSA(right)
    x = clean_params[0]
    self.saveBrainCol(right, x)
    for k in range(0, len(self.Brain[right][x])):
      if k != params[2]:
        self.Brain[right][x][k] = ((1 - Agent.lambda_const * self.Brain[right][x][k]) \
                                  / (1 - self.Brain[right][x][k])) \
                                  * self.Brain[right][x][k]
      else:
        self.Brain[right][x][k] *= Agent.lambda_const
    if any(elem < Agent.MinProb for elem in self.Brain[right][x]):
      assert len(self.Brain[right][x]) == len(self.Stack[right][-1][2][x])
      for k in range(0, len(self.Brain[right][x])):
        self.Brain[right][x][k] = self.Stack[right][-1][2][x][k]
    assert all(all(elem >= Agent.MinProb for elem in self.Brain[right][x]) for x in range(0, Agent.m))

  def MoveDist(self, right, params, clean_params):
    self.SSA(right)
    x, y = clean_params[0], clean_params[1]
    self.saveBrainCol(right, x)
    for k in range(0, len(self.Brain[right][x])):
      self.Brain[right][x][k] = self.Brain[right][y][k]
    if any(elem < Agent.MinProb for elem in self.Brain[right][x]):
      assert len(self.Brain[right][x]) == len(self.Stack[right][-1][2][x])
      for k in range(0, len(self.Brain[right][x])):
        self.Brain[right][x][k] = self.Stack[right][-1][2][x][k]
    assert all(all(elem >= Agent.MinProb for elem in self.Brain[right][x]) for x in range(0, Agent.m))

  def ins_IncProbLeft(self, params, clean_params):
    self.IncProb(False, params, clean_params)
  def ins_IncProbRight(self, params, clean_params):
    self.IncProb(True, params, clean_params)

  def ins_DecProbLeft(self, params, clean_params):
    self.DecProb(False, params, clean_params)
  def ins_DecProbRight(self, params, clean_params):
    self.DecProb(True, params, clean_params)

  def ins_MoveDistLeft(self, params, clean_params):
    self.MoveDist(False, params, clean_params)
  def ins_MoveDistRight(self, params, clean_params):
    self.MoveDist(True, params, clean_params)

  def ins_IncProbBoth(self, params, clean_params):
    if random.randint(0, 1):
      self.ins_IncProbLeft(params, clean_params)
      self.ins_IncProbRight(params, clean_params)
    else:
      self.ins_IncProbRight(params, clean_params)
      self.ins_IncProbLeft(params, clean_params)
  def ins_DecProbBoth(self, params, clean_params):
    if random.randint(0, 1):
      self.ins_DecProbLeft(params, clean_params)
      self.ins_DecProbRight(params, clean_params)
    else:
      self.ins_DecProbRight(params, clean_params)
      self.ins_DecProbLeft(params, clean_params)
  def ins_SSAandCopy(self, params, clean_params):
    if params[0] >= 5:
      return
    self.BlockSSA[0], self.BlockSSA[1] = False, False
    if random.randint(0, 1):
      self.SSA(False)
      self.SSA(True)
    else:
      self.SSA(True)
      self.SSA(False)
    t = self.time
    [tl, rl] = self.Stack[0][-1][:2] if len(self.Stack[0]) > 0 else [0, 0]
    [tr, rr] = self.Stack[1][-1][:2] if len(self.Stack[1]) > 0 else [0, 0]

    loser = (self.IR[0] - rl) / (tl - t) > (self.IR[1] - rr) / (tr - t)
    def diff(xs, ys):
      return sum(abs(x - y) for x, y in zip(xs, ys)) != 0
    for x in range(0, Agent.m):
      if diff(self.Brain[0][x], self.Brain[1][x]):
        self.saveBrainCol(loser, x)
        for k in range(0, Agent.n):
          self.Brain[loser][x][k] = self.Brain[not loser][x][k]



  
  def updateInputs(self):
    x, y, d = self.State[0], self.State[1], self.State[2] / 100 * (2 * math.pi)
    front_v = [[x, y], [24 * (x + math.cos(d)), 24 * (y + math.sin(d))]]
    self.State[3] = 24 if any(intersect(wall, front_v) for wall in self.walls) else 0

    right_v  = [[x, y], [24 * (x + math.cos(d - math.pi / 2)), 24 * (y + math.sin(d - math.pi / 2))]]
    self.State[4] = 24 if any(intersect(wall, right_v) for wall in self.walls) else 0

    back_v = [[x, y], [-24 * (x + math.cos(d)), -24 * (y + math.sin(d))]]
    self.State[5] = 24 if any(intersect(wall, back_v) for wall in self.walls) else 0

    left_v  = [[x, y], [24 * (x + math.cos(d + math.pi / 2)), 24 * (y + math.sin(d + math.pi / 2))]]
    self.State[6] = 24 if any(intersect(wall, left_v) for wall in self.walls) else 0



  def ins_MoveAgent(self, params, clean_params):
    Vel = 10 
    x, y, d = self.State[0], self.State[1], self.State[2] / 100 * (2 * math.pi)
    move = [[x, y], [x + Vel * math.cos(d), y + Vel * math.sin(d)]]
    assert abs((move[1][0] - x) ** 2 + (move[1][1] - y) ** 2 - Vel ** 2) < 1e-6
    nx, ny = int(move[1][0]), int(move[1][1])
    move[1][0] += 30 * math.cos(d)
    move[1][1] += 30 * math.sin(d)
    if not any(intersect(wall, move) for wall in self.walls):
      self.State[0], self.State[1] = nx, ny
    self.updateInputs()
  def ins_SetDirection(self, params, clean_params):
    self.State[2] = params[0] / Agent.n * 100
    self.updateInputs()

  tab_ins = [
      ["Jmpl", 6], ["Jmpeq", 6],
      ["Add", 6], ["Sub", 6], ["Mul", 6], ["Div", 6],
      ["Mov", 4], ["Init", 4],
      ["Bet", 4],
      ["GetLeft", 3], ["GetRight", 3],
      ["EnableSSALeft", 1], ["EnableSSARight", 1],
      ["IncProbLeft", 3], ["DecProbLeft", 3], ["MoveDistLeft", 4],
      ["IncProbRight", 3], ["DecProbRight", 3], ["MoveDistRight", 4],
      ["IncProbBoth", 3], ["DecProbBoth", 3], ["SSAandCopy", 1],
      ["MoveAgent", 0], ["SetDirection", 1]
  ]
          

  def exec_ins(self, ins_idx, params):
    ins_name = 'ins_' + Agent.tab_ins[ins_idx][0]
    assert Agent.tab_ins[ins_idx][1] == len(params) or (Agent.tab_ins[ins_idx][0] == "Bet" and len(params) == 6)
    ins_method = getattr(self, ins_name)
    clean_params = []
    for i in range(0, len(params) - 1, 2):
      assert i + 1 < len(params) 
      clean_params.append((params[i] * Agent.n + params[i+1]) % Agent.m)
    assert len(clean_params) == len(params) // 2
    ins_method(params, clean_params)

  def getDecision(self, idx):
    def f(x, y):
      return x * y
    def g(idx):
      d = dict()
      d["GetLeft"] = "GetRight"
      d["GetRight"] = "GetLeft"
      d["EnableSSALeft"] = "EnableSSARight"
      d["EnableSSARight"] = "EnableSSALeft"
      d["IncProbLeft"] = "IncProbRight"
      d["IncProbRight"] = "IncProbLeft"
      d["DecProbLeft"] = "DecProbRight"
      d["DecProbRight"] = "DecProbLeft"
      d["MoveDistRight"] = "MoveDistLeft"
      d["MoveDistLeft"] = "MoveDistRight"
      anti = d.get(Agent.tab_ins[idx][0], Agent.tab_ins[idx][0])
      for i, val in enumerate(Agent.tab_ins):
        if val[0] == anti:
          return i 
    def Q(i, j):
      return f(self.Brain[0][i][j], self.Brain[1][i][g(j)]) / sum(f(self.Brain[0][i][j], self.Brain[1][i][g(j)]) for j in range(0, Agent.n))
    nums = list(range(0, Agent.n))
    weights = [Q(idx, j) for j in nums]
    return random.choices(nums, weights)[0]
  def act(self):
    
    # select instruction head a[j] with max? probability Q(IP, j)
    ins_idx = self.getDecision(self.InsPtr)
    self.time += 1
    self.InsStat[Agent.tab_ins[ins_idx][0]] += 1

    # select arguments 
    params = []
    for i in range(1, Agent.tab_ins[ins_idx][1] + 1):
      param = self.getDecision(self.InsPtr + i)
      self.time += 1
      params.append(param)
    # print("exec %s with args %s" % (Agent.tab_ins[ins_idx][0], params))

    # take care of Bet!
    if Agent.tab_ins[ins_idx][0] == "Bet":
      nums = list(range(0, Agent.n))
      weights = [(self.Brain[0][self.InsPtr + 5][j] / sum(self.Brain[0][self.InsPtr + 5])) for j in nums]
      c = random.choices(nums, weights)[0]
      self.time += 1
      c = 1 if c > (Agent.n / 2) else -1
      params.append(c)

      weights = [(self.Brain[1][self.InsPtr + 5][j] / sum(self.Brain[1][self.InsPtr + 5])) for j in nums]
      d = random.choices(nums, weights)[0]
      self.time += 1
      d = 1 if d > (Agent.n / 2) else -1
      params.append(d)
    
     
    self.exec_ins(ins_idx, params)

    # external reward
    if self.ER != 0:
      self.State[8] = self.ER

    # TODO if an input has changed S0-S8, then shift S0-S80 to S9-S89
    
    if Agent.tab_ins[ins_idx][0] != "Jmpl" and Agent.tab_ins[ins_idx][0] != "Jmpeq":
      w1 = self.getDecision(self.InsPtr + 7)
      w2 = self.getDecision(self.InsPtr + 8)
      self.time += 2
      w = (w1 * Agent.n + w2) % Agent.m
      self.InsPtr = w - (w % Agent.InsBlockSize)
    assert all(all(elem >= Agent.MinProb for elem in self.Brain[0][x]) for x in range(0, Agent.m))
    assert all(all(elem >= Agent.MinProb for elem in self.Brain[1][x]) for x in range(0, Agent.m))
