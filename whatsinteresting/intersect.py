# https://algorithmtutor.com/Computational-Geometry/Determining-if-two-consecutive-segments-turn-left-or-right/
class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y
  def subtract(self, p):
    return Point(self.x - p.x, self.y - p.y)
  def __str__(self):
    return '(' + str(self.x) + ', ' + str(self.y) + ')'

def cross_product(p1, p2):
  return p1.x * p2.y - p2.x * p1.y

def direction(p1, p2, p3):
  return cross_product(p3.subtract(p1), p2.subtract(p1))
def on_segment(p1, p2, p):
  return min(p1.x, p2.x) <= p.x <= max(p1.x, p2.x) and min(p1.y, p2.y) <= p.y <= max(p1.y, p2.y)

def intersect(p12, p34):
  p1 = Point(p12[0][0], p12[0][1])
  p2 = Point(p12[1][0], p12[1][1])
  p3 = Point(p34[0][0], p34[0][1])
  p4 = Point(p34[1][0], p34[1][1])
  print(p1)
  print(p2)
  print(p3)
  print(p4)

  d1 = direction(p3, p4, p1)
  d2 = direction(p3, p4, p2)
  d3 = direction(p1, p2, p3)
  d4 = direction(p1, p2, p4)

  if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
      ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
    return True

  elif d1 == 0 and on_segment(p3, p4, p1):
    return True
  elif d2 == 0 and on_segment(p3, p4, p2):
    return True
  elif d3 == 0 and on_segment(p1, p2, p3):
    return True
  elif d4 == 0 and on_segment(p1, p2, p4):
    return True
  else:
    return False
