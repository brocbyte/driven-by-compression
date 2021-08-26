#!/usr/bin/env python3
from interest import Agent
lw = 100
rw = 900
walls = [
  [[lw, lw], [lw, rw]],
  [[lw, lw], [rw, lw]],
  [[rw, lw], [rw, rw]],
  [[lw, rw], [rw, rw]]
]

agent = Agent(walls)
import tkinter as tk
root = tk.Tk()
delta_t = 20
size = 1000
canvas = tk.Canvas(root, width = size, height = size)
for wall in walls:
  canvas.create_line(wall[0][0], wall[0][1], wall[1][0], wall[1][1], dash=(4, 2))
canvas.pack()

dot_size = 8 
x, y = agent.State[0], agent.State[1]
dot = canvas.create_oval(x, y, x + dot_size, y + dot_size, outline="white", fill="green") 

lastTime = 0
def redraw():
  global x, y
  global lastTime
  if agent.time - lastTime >= 10000:
    print("# " + f'{agent.time:,}')
    print("SL: %d, SR: %d" % (len(agent.Stack[0]), len(agent.Stack[1])))
    print("RL: %d, RR: %d" % (agent.IR[0], agent.IR[1]))
    print(sorted(agent.InsStat.items(), key=lambda item: item[1]))
    lastTime = agent.time

  # print("(%d ; %d)" % (x, y))
  agent.act()
  nx, ny = agent.State[0], agent.State[1]
  canvas.create_oval(x, y, x + dot_size, y + dot_size, outline="white", fill="green") 
  canvas.move(dot, nx - x, ny - y)
  x, y = nx, ny
  #canvas.update()
  canvas.after(delta_t,redraw)
 
'''
steps = int(1e6)
for i in range(0, steps):
  agent.act()

print("%d steps passed!" % steps)
'''
canvas.after(delta_t, redraw)
root.mainloop()
