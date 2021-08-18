#!/usr/bin/env python3
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

while True:
  # select instruction head a[j] with probability Q(IP, j)
  # select arguments 
  # execute the instruction

