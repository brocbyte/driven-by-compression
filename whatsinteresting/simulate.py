#!/usr/bin/env python3
from interest import Agent
agent = Agent()
import tkinter as tk
root = tk.Tk()
delta_t = 20
size = 1000
canvas = tk.Canvas(root, width = size, height = size)
canvas.pack()

x, y = 0, 0
dot = canvas.create_oval(x,y,x+10,y+10,outline="white",fill="green") 

def redraw():
  canvas.after(delta_t,redraw)
  agent.act()
  nx, ny = agent.State[0], agent.State[1]
  canvas.move(dot, nx - x, ny - y)
  canvas.update()

canvas.after(delta_t, redraw)
root.mainloop()
