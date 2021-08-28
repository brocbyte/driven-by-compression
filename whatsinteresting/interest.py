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
  random.seed(100)
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
  def __init__(self, walls = []):
    self.State = [0] * Agent.m
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
    self.pos = [500.0, 500.0]
    self.direction = 0.0
    self.time = 0
    # stack elements look like that:
    # [t, RL(t), (c1, Left(c1)), ...]
    # where t - checkpoint time
    #       RL(t) - reward until time t
    #       c1 - idx of modified column
    #       Left(c1) - previous Left-column

    self.BlockSSA = [False] * 2
    self.NewInput = False

    self.BS = [i * Agent.InsBlockSize for i in range(Agent.m) if i * Agent.InsBlockSize < Agent.m]
    self.noBS = [x for x in range(Agent.m) if x not in self.BS]
    self.Q = np.zeros((Agent.m, Agent.n))

    

    self.InsStat = [0] * Agent.n
    self.InsTrack = [[] for _ in range(Agent.n)]
    self.RewardTrack = [[], []]
    self.StackTrack = [[], []]
    self.path = []

    Agent.name_to_idx = dict()
    for idx, ins in enumerate(Agent.tab_ins):
      Agent.name_to_idx[ins[0]] = idx

  # SSA Calls
  # https://people.idsia.ch/~juergen/mljssalevin/node2.html
  def SSA(self, right):
    if not self.BlockSSA[right]:
      self.BlockSSA[right] = True

      # t is a new checkpoint
      t = self.time

      while self.Stack[right]:
        # t' and t''
        [t1, IRt1] = self.Stack[right][-1][:2]
        [t2, IRt2] = self.Stack[right][-2][:2] if len(self.Stack[right]) >= 2 else [0, 0]

        # main induct rule
        assert t != t2
        assert t != t1
        if (self.IR[right] - IRt1) / (t - t1) > (self.IR[right] - IRt2) / (t - t2):
          # we're cool!
          break
        else:
          # pop t1 block, restore everything as it was before t1
          lblock = self.Stack[right].pop()
          modifs = lblock[2]
          for k, v in modifs.items():
            for i in range(0, len(self.Brain[right][k])):
              self.Brain[right][k][i] = v[i]

      self.Stack[right].append([t, self.IR[right], dict()])

  def ins_Jmpl(self, params, clean_params):
    if self.State[clean_params[0]] < self.State[clean_params[1]]:
      self.InsPtr = clean_params[2] - (clean_params[2] % Agent.InsBlockSize)
  def ins_Jmpeq(self, params, clean_params):
    if self.State[clean_params[0]] == self.State[clean_params[1]]:
      self.InsPtr = clean_params[2] - (clean_params[2] % Agent.InsBlockSize)

  def ins_Add(self, params, clean_params):
    self.State[clean_params[2]] = (self.State[clean_params[0]] + self.State[clean_params[1]]) % Agent.M
  def ins_Sub(self, params, clean_params):
    self.State[clean_params[2]] = (self.State[clean_params[0]] - self.State[clean_params[1]]) % Agent.M
  def ins_Mul(self, params, clean_params):
    self.State[clean_params[2]] = (self.State[clean_params[0]] * self.State[clean_params[1]]) % Agent.M
  def ins_Div(self, params, clean_params):
    if self.State[clean_params[1]] != 0:
      self.State[clean_params[2]] = (self.State[clean_params[0]] // self.State[clean_params[1]]) % Agent.M

  def ins_Mov(self, params, clean_params):
    self.State[clean_params[1]] = self.State[clean_params[0]]
  def ins_Init(self, params, clean_params):
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
    self.NewInput = True

  def ins_GetLeft(self, params, clean_params):
    self.State[clean_params[0]] = math.ceil(Agent.M * self.Brain[0][clean_params[0]][params[2]])
  def ins_GetRight(self, params, clean_params):
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
      self.Stack[right][-1][2][column] = self.Brain[right][column][:]
    
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
      for k in range(0, len(self.Brain[right][x])):
        self.Brain[right][x][k] = self.Stack[right][-1][2][x][k]

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
      for k in range(0, len(self.Brain[right][x])):
        self.Brain[right][x][k] = self.Stack[right][-1][2][x][k]

  def MoveDist(self, right, params, clean_params):
    self.SSA(right)
    x, y = clean_params[0], clean_params[1]
    self.saveBrainCol(right, x)
    for k in range(0, len(self.Brain[right][x])):
      self.Brain[right][x][k] = self.Brain[right][y][k]

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
    if t == tl or t == tr:
      return
    loser = (self.IR[0] - rl) / (t - tl) - (self.IR[1] - rr) / (t - tr)
    if loser == 0:
      return
    loser = loser > 0
    def diff(xs, ys):
      return sum(abs(x - y) for x, y in zip(xs, ys)) != 0
    for x in range(0, Agent.m):
      if diff(self.Brain[0][x], self.Brain[1][x]):
        self.saveBrainCol(loser, x)
        for k in range(0, Agent.n):
          self.Brain[loser][x][k] = self.Brain[not loser][x][k]



  
  def updateInputs(self):
    x, y, d = self.pos[0], self.pos[1], self.direction

    self.State[0] = np.round(x)
    self.State[1] = np.round(y)
    self.State[2] = np.round(d / (2 * math.pi) * 100)

    front_v = [[x, y], [24 * (x + math.cos(d)), 24 * (y + math.sin(d))]]
    self.State[3] = 24 if any(intersect(wall, front_v) for wall in self.walls) else 0

    right_v  = [[x, y], [24 * (x + math.cos(d - math.pi / 2)), 24 * (y + math.sin(d - math.pi / 2))]]
    self.State[4] = 24 if any(intersect(wall, right_v) for wall in self.walls) else 0

    back_v = [[x, y], [-24 * (x + math.cos(d)), -24 * (y + math.sin(d))]]
    self.State[5] = 24 if any(intersect(wall, back_v) for wall in self.walls) else 0

    left_v  = [[x, y], [24 * (x + math.cos(d + math.pi / 2)), 24 * (y + math.sin(d + math.pi / 2))]]
    self.State[6] = 24 if any(intersect(wall, left_v) for wall in self.walls) else 0

    self.NewInput = True



  def ins_MoveAgent(self, params, clean_params):
    Vel = 10 
    x, y, d = self.pos[0], self.pos[1], self.direction
    move = [[x, y], [x + Vel * math.cos(d), y + Vel * math.sin(d)]]
    assert abs((move[1][0] - x) ** 2 + (move[1][1] - y) ** 2 - Vel ** 2) < 1e-6
    nx, ny = int(move[1][0]), int(move[1][1])
    move[1][0] += 30 * math.cos(d)
    move[1][1] += 30 * math.sin(d)
    if not any(intersect(wall, move) for wall in self.walls):
      self.pos = [nx, ny]
    self.updateInputs()
    self.NewInput = True

  def ins_SetDirection(self, params, clean_params):
    self.direction = params[0] / Agent.n * (2 * math.pi)
    self.updateInputs()
    self.NewInput = True

  
          

  def exec_ins(self, ins_idx, params):
    ins_name = 'ins_' + Agent.tab_ins[ins_idx][0]
    ins_method = getattr(self, ins_name)
    clean_params = []
    for i in range(0, len(params) - 1, 2):
      clean_params.append((params[i] * Agent.n + params[i+1]) % Agent.m)
    ins_method(params, clean_params)

  
  def updateQ(self):
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
    def f(x, y):
      return x * y
    for i in self.noBS:
      for j in range(Agent.n):
        self.Q[i][j] = f(self.Brain[1][i][j], self.Brain[1][i][j])
      normalizer = sum(self.Q[i])
      self.Q[i] /= normalizer
    for i in self.BS:
      for j in range(Agent.n):
        self.Q[i][j] = f(self.Brain[0][i][j], self.Brain[1][i][g(j)])
      normalizer = sum(self.Q[i])
      self.Q[i] /= normalizer
      
  def getDecision(self, idx):
    return np.random.choice(Agent.n, 1, p = self.Q[idx])[0]
    
  def act(self):
    for i in range(2):
      self.StackTrack[i].append(len(self.Stack[i]))
      self.RewardTrack[i].append(self.IR[i])
    for i in range(Agent.n):
      self.InsTrack[i].append(self.InsStat[i])
    self.path.append(self.pos)
    self.time += 1

    # select instruction head a[j]
    self.updateQ() 
    ins_idx = self.getDecision(self.InsPtr)
    self.InsStat[ins_idx] += 1

    # select arguments 
    params = []
    for i in range(Agent.tab_ins[ins_idx][1]):
      param = self.getDecision(self.InsPtr + i + 1)
      params.append(param)
    # print("exec %s with args %s" % (Agent.tab_ins[ins_idx][0], params))

    # take care of Bet!
    if Agent.tab_ins[ins_idx][0] == "Bet":
      pc = self.Brain[0][self.InsPtr + 5]
      pc = pc / np.sum(pc)
      c = np.random.choice(Agent.n, 1, p = pc)[0]
      c = 1 if c > (Agent.n / 2) else -1
      params.append(c)

      pd = self.Brain[1][self.InsPtr + 5]
      pd = pd / np.sum(pd)
      d = np.random.choice(Agent.n, 1, p = pd)[0]
      d = 1 if d > (Agent.n / 2) else -1
      params.append(d)
    
     
    self.exec_ins(ins_idx, params)

    # external reward
    if self.ER != 0:
        self.State[8] = self.ER
        self.NewInput = True

    # if an input has changed S0-S8, then shift S0-S80 to S9-S89
    if self.NewInput:
      for i in range(89, 8, -1):
        self.State[i] = self.State[i - 9]
      self.NewInput = False
    
    if Agent.tab_ins[ins_idx][0] != "Jmpl" and Agent.tab_ins[ins_idx][0] != "Jmpeq":
      w1 = self.getDecision(self.InsPtr + 7)
      w2 = self.getDecision(self.InsPtr + 8)
      w = (w1 * Agent.n + w2) % Agent.m
      self.InsPtr = w - (w % Agent.InsBlockSize)
