#!/usr/bin/env python3
from interest import Agent
import numpy as np
import matplotlib.pyplot as plt
import sys
lw = 100
rw = 900
walls = [
  [[lw, lw], [lw, rw]],
  [[lw, lw], [rw, lw]],
  [[rw, lw], [rw, rw]],
  [[lw, rw], [rw, rw]]
]
# act and record everything
agent = Agent(walls)

T = int(sys.argv[1])
while agent.time < T:
  agent.act()

# draw everything

import tkinter as tk
root = tk.Tk()
delta_t = 1 
size = 1000
canvas = tk.Canvas(root, width = size, height = size)

for wall in walls:
  canvas.create_line(wall[0][0], wall[0][1], wall[1][0], wall[1][1], dash=(4, 2))
canvas.pack()

dot_size = 8 

i = 0
[x, y] = agent.path[i]
dot = canvas.create_oval(x, y, x + dot_size, y + dot_size, outline="white", fill="green") 


# plot reward evolution of each modulo
LEFT_reward_track = np.array(agent.RewardTrack[0])
RIGHT_reward_track  = np.array(agent.RewardTrack[1])

plt.xlim(-.2, T)
plt.ylim(min(min(LEFT_reward_track),min(RIGHT_reward_track)),max(max(LEFT_reward_track),max(RIGHT_reward_track)))

LEFT, = plt.plot(LEFT_reward_track,'-')
RIGHT, = plt.plot(RIGHT_reward_track,'--')
plt.legend([LEFT, RIGHT], ['LEFT Reward', 'RIGHT Reward'])
plt.show()

LEFT_stack_track = np.array(agent.StackTrack[0])
RIGHT_stack_track  = np.array(agent.StackTrack[1])

plt.xlim(-.2, T)
plt.ylim(min(min(LEFT_stack_track),min(RIGHT_stack_track)),max(max(LEFT_stack_track),max(RIGHT_stack_track)))

LEFT, = plt.plot(LEFT_stack_track,'-')
RIGHT, = plt.plot(RIGHT_stack_track,'--')
plt.legend([LEFT, RIGHT], ['LEFT Stack', 'RIGHT Stack'])
plt.show()


def redraw():
  global x, y, T
  global i
  [nx, ny] = agent.path[i]
  i = (i + 1) % T
  canvas.create_oval(x, y, x + dot_size, y + dot_size, outline="white", fill="green") 
  canvas.move(dot, nx - x, ny - y)
  x, y = nx, ny
  canvas.after(delta_t,redraw)
 
canvas.after(delta_t, redraw)
root.mainloop()
