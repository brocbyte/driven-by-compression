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
dot = canvas.create_oval(x,y,x+20,y+20,outline="white",fill="green") 

def redraw():
  canvas.after(delta_t,redraw)
  canvas.move(dot, 5, 5)
  canvas.update()

canvas.after(delta_t, redraw)
root.mainloop()
