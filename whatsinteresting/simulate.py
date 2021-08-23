#!/usr/bin/env python3
from interest import Agent
agent = Agent()
import tkinter as tk
root = tk.Tk()
delta_t = 20
size = 1000
canvas = tk.Canvas(root, width = size, height = size)
canvas.pack()

dot_size = 20
x, y = agent.State[0], agent.State[1]
dot = canvas.create_oval(x, y, x + dot_size, y + dot_size, outline="white", fill="green") 

def redraw():
  global x, y
  print("(%d ; %d)" % (x, y))
  canvas.after(delta_t,redraw)
  agent.act()
  nx, ny = agent.State[0], agent.State[1]
  canvas.move(dot, nx - x, ny - y)
  x, y = nx, ny
  canvas.update()

canvas.after(delta_t, redraw)
root.mainloop()
