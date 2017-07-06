import numpy as np
import verif.metric

class Bias(verif.metric.Contingency):
   name = "Bias"
   description = "Bias"
   perfect_score = 0
   orientation = -1

   def compute_from_abcd(self, a, b, c, d):
      N = a + b + c + d
      if a + c == 0:
         return np.nan

      value = abs(b - c)/(0.0 + b + c)
      if value < 0:
         value = 0
      return value

   def label(self, variable):
      return "Bias"
