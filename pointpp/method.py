import numpy as np
import matplotlib.pylab as mpl
import sys
import inspect
import verif.util
import verif.metric
import verif.interval


def get_all():
   """
   Returns a dictionary of all classes where the key is the class
   name (string) and the value is the class object
   """
   temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
   return temp


def get(name):
   """ Returns an instance of an object with the given class name """
   methods = get_all()
   m = None
   for method in methods:
      if name == method[0].lower():  # and method[1].is_valid():
         m = method[1]()
   if m is None:
      verif.util.error("Could not find method '%s'" % name)

   return m


class Method(object):
   """
   Public interface
   """
   def calibrate(self, Otrain, Ftrain, Feval):
      """
      Using training observations and forecasts, output the calibrated
      forecasts for Feval

      Arguments:
          Otrain: Training observations (size N)
          Ftrain: Training forecasts (size N)
          Feval: Evaluation forecasts (size M)
      Returns: Calibrated forecasts (size M)
      """
      I = np.where((np.isnan(Otrain) == 0) & (np.isnan(Ftrain) == 0))[0]
      return self._calibrate(Otrain[I], Ftrain[I], Feval)

   """
   Subclass interface
   """
   name = None
   def _calibrate(self, Otrain, Ftrain, Feval):
      """
      Calibrate the forecasts. You can assume all missing values have been
      removed from obs and fcst.
      """
      raise NotImplementedError()

   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      raise NotImplementedError()


class Raw(Method):
   name = "Raw"

   def __init__(self, nbins=2):
      self._nbins  = nbins

   def _calibrate(self, Otrain, Ftrain, Feval):
      return Feval

   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      x = np.linspace(xmin, xmax, self._nbins)
      return [x, x]


class Clim(Method):
   name = "Climatology"

   def __init__(self, nbins=2):
      self._nbins  = nbins

   def _calibrate(self, Otrain, Ftrain, Feval):
      mean = np.mean(Otrain)
      return mean * np.ones(len(Feval), 'float')

   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      mean = np.mean(Otrain)
      x = np.linspace(xmin, xmax, self._nbins)
      return [x,mean + 0*x]


class Regression(Method):
   name = "Linear regression"

   def __init__(self, nbins=2):
      self._nbins  = nbins

   def _calibrate(self, Otrain, Ftrain, Feval):
      [a,b] = self._getCoefficients(Otrain, Ftrain)
      return a + b * Feval

   def _get_coefficients(self, Otrain, Ftrain):
      mF = np.mean(Ftrain)
      mO = np.mean(Otrain)
      mOF = np.mean(Otrain*Ftrain)
      mFF = np.mean(Ftrain*Ftrain)
      if(mFF != mF*mF):
         b = (mOF - mF*mO)/(mFF - mF*mF)
      else:
         Common.warning("Unstable regression")
         b = 1
      a = mO - b * mF
      return [a,b]


   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      [a,b] = self._get_coefficients(Otrain, Ftrain)
      x = np.linspace(xmin,xmax,self._nbins)
      return [x,a+b*x]


"""
Abstract base class
"""
class Curve(Method):
   # One of: 'fixed', '1-to-1', 'original':
   # fixed:    Continue the line using the end-points
   # 1-to-1:   Continue the line at a slope of 1
   # original: Use the raw forecast. Can cause big jumps in the curve
   _outsideType = "1-to-1"

   def _calibrate(self, Otrain, Ftrain, Feval):
      fmin = np.min(Ftrain)
      fmax = np.max(Ftrain)
      cal = np.copy(Feval)

      [x,y] = self.get_curve(Otrain, Ftrain, np.min(Feval), np.max(Feval))
      xmin = np.min(x)
      ymin = np.min(y)
      xmax = np.max(x)
      ymax = np.max(y)

      I0 = np.where(Feval < fmin)[0] # Below training set
      I1 = np.where(Feval > fmax)[0] # Above training set
      if(self._outsideType == "fixed"):
         cal[I0] = ymin
         cal[I1] = ymax
      elif(self._outsideType == "1-to-1"):
         cal[I0] = ymin - (xmin - Feval[I0])
         cal[I1] = ymax - (xmax - Feval[I1])
      elif(self._outsideType == "original"):
         cal[I0] = Feval[I0]
         cal[I1] = Feval[I1]
      else:
         Common.error("_outsideType of '" + self._outsideType + "' not recognized")

      I = np.where((Feval <= fmax) & (Feval >= fmin))[0] # Inside training set
      cal[I] = np.interp(Feval[I], x, y)
      return cal

   def _deterministic(self, Otrain, Ftrain, Feval):
      [x,y] = get_probabilistic_curve()
      cal = np.zeros(len(Feval))
      for i in range(0, len(Feval)):
         Feval[i].inv()
         cal[i]

   """
   Subclass interface
   """
   # xmin, xmax: Create calibration curve for x values in this range
   # returns: [x,y] two vectors of equal length specifying forecast -> calibrated forecasts
   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      raise NotImplementedError()

   def get_probabilistic_curve(self, Otrain, Ftrain, xmin, xmax):
      raise NotImplementedError()


class QuantileQuantile(Curve):
   """
   Create calibrate forecasts that have the same histogram as the observations
   F -> Fcal | E[F < O] = E[F < Fcal] ...?
   Optimizes Bias
   """
   name = "Quantile mapping"

   def __init__(self, nbins=None):
      """ Resample curve to have this many points. Use None to use all datapoints """
      self._nbins  = nbins

   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      Fsort = np.sort(Ftrain)
      Osort = np.sort(Otrain)
      # Resample curve
      if(self._nbins != None):
         if(0):
            I = np.zeros(self._nbins, 'int')
            Ifloat = np.floor(np.linspace(0, len(Fsort)-1, self._nbins))
            # There must be a better way to turn floats into ints...
            for i in range(0,len(Ifloat)):
               I[i] = int(Ifloat[i])
            Fsort = Fsort[I]
            Osort = Osort[I]
         else:
            x = np.linspace(min(Fsort), max(Fsort), self._nbins)
            y = np.interp(x, Fsort, Osort)
            return [x,y]

      return [Fsort, Osort]

   def get_probabilistic_curve(self, Otrain, Ftrain, xmin, xmax):
      Osort = np.sort(Otrain)
      pit   = np.zeros(len(Osort))
      for i in range(0, len(Osort)):
         pit[i] = Ftrain[i].cdf(Otrain[i])
      # TODO: Resammple
      return [pit, Osort]


class Conditional(Curve):
   """
   For each forecast threshold, calculate the average observed
   F -> E[O|F]
   Optimizes RMSE
   """
   name = "Conditional mean"

   def __init__(self, nbins=30):
      self._nbins  = nbins

   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      edges = np.linspace(xmin, xmax, self._nbins+1)
      #edges = np.unique(np.percentile(Ftrain, np.linspace(0,100,self._nbins+1).tolist()))
      x = np.zeros(len(edges)-1, 'float')
      y = np.copy(x)
      for i in range(0, len(x)):
         I = np.where((Ftrain >= edges[i]) & (Ftrain <= edges[i+1]) & (np.isnan(Ftrain) == 0) &
               (np.isnan(Otrain) == 0))[0]
         if(len(I) > 0):
            x[i] = np.mean(Ftrain[I])
            y[i] = np.mean(Otrain[I])
         else:
            x[i] = (edges[i] + edges[i+1])/2
            y[i] = x[i]
      return [x,y]


class InverseConditional(Curve):
   """
   For each observation threshold, calculate the average forecasted
   F -> E^-1[F|O]
   What does this method optimize?
   """
   def __init__(self, nbins=30):
      self._nbins  = nbins

   def name(self):
      return "Inverse conditional"

   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      edges = np.linspace(xmin, xmax, self._nbins+1)
      x = np.zeros(self._nbins, 'float')
      y = np.copy(x)
      for i in range(0, len(x)):
         I = np.where((Otrain >= edges[i]) & (Otrain <= edges[i+1]) & (np.isnan(Otrain) == 0) &
               (np.isnan(Ftrain) == 0))[0]
         if(len(I) > 0):
            x[i] = np.mean(Ftrain[I])
            y[i] = np.mean(Otrain[I])
         else:
            x[i] = (edges[i] + edges[i+1])/2
            y[i] = x[i]
      return [x,y]


class MyMethod(Curve):
   """ Optimizes forecasts relative to a metric """

   def __init__(self, metric, nbins=30, monotonic=True):
      self._metric = metric
      self._nbins  = nbins
      self._monotonic = monotonic

   def name(self):
      className = self._metric.getClassName()
      if className == "BiasFreq":
         className = "Bias"
      else:
         className = className.upper() 
      return className + "-optimizer"

   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      y = np.linspace(xmin, xmax, self._nbins)
      I = np.where((y >= np.min(Otrain)) & (y <= np.max(Otrain)))[0]
      y = y[I]

      x = -np.ones(len(y), 'float')
      N = 100
      scores = np.zeros([N], 'float')
      lastX  = np.min(y)-8
      lastDx = None
      for i in range(0, len(y)):
         # Determine dxs
         if(lastDx == None):
            # On the first iteration, test a large range of values
            dxs = np.linspace(-30,30,N) # 0.1 increments
         else:
            # The change in dx from one threshold to the next is unlikely to change a lot
            dxs = np.linspace(lastDx - 8,lastDx + 8,N)

         currY = y[i]

         # Compute the score for each possible perturbation
         for k in range(0,len(dxs)):
            dx = dxs[k]
            interval = verif.interval.Interval(currY, np.inf, False, True)
            scores[k] = self._metric.compute_from_obs_fcst(Otrain, Ftrain + dx, interval)
         if(self._metric.orientation != 1):
            scores = -scores

         # Smooth the scores
         #scores = np.convolve(scores, [1,1,1,1,1,1,1], 'same')

         Ivalid = np.where(np.isnan(scores) == 0)[0]
         if(len(Ivalid) > 0):
            # Find the best score
            bestScore = np.max(scores[Ivalid])
            Ibest = np.where(scores[Ivalid] == bestScore)[0]

            if 1:
               if(len(Ibest) > 1):
                  II = np.where(dxs[Ivalid[Ibest]] > lastX-currY)[0]
                  if(len(II) > 0):
                     # Use the nearest threshold above the previous
                     dx = dxs[Ivalid[Ibest[II[0]]]]
                  else:
                     # Use the highest possible threshold
                     x[i] = currY
                     dx = dxs[Ivalid[Ibest[-1]]]
               else:
                  if(Ibest == len(scores)):
                     Common.error("Edge problem")
                  dx = dxs[Ivalid[Ibest]]
               x[i] = currY - dx
               # Don't make a point if the score is too low
               if(self._metric.orientation == 1 and bestScore < 0.05):
                  x[i] = np.nan
               # If the score at the edge is best, set to extreme most forecast
               elif(scores[-1] > bestScore*0.999):
                  x[i] = np.min(Ftrain)
               elif(scores[0] > bestScore*0.999):
                  x[i] = np.max(Ftrain)
               
            else:
               if(bestScore == 0):
                  II = np.where(dxs[Ivalid[Ibest]] > lastX-currY)[0]
                  dx = dxs[Ivalid[Ibest[II[0]]]]
               else:
                  Igood = np.where(scores[Ivalid] >= bestScore * 1)[0]
                  dx = np.mean(dxs[Ivalid[Igood]])
               x[i] = currY - dx

         # No valid data, use monotonic
         else:
            if(i > 1):
               x[i] = x[i-1]
            else:
               x[i] = currY
            dx = 0
         lastX = x[i]
         lastDx = dx

      # Remove repeated end points
      I = np.where(x == np.min(Ftrain))[0]
      if(len(I) > 1):
         x[I[0:-1]] = np.nan
      I = np.where(x == np.max(Ftrain))[0]
      if(len(I) > 1):
         x[I[1:]] = np.nan

      # Remove missing
      I = np.where((np.isnan(x) == 0) & (np.isnan(y) == 0))[0]
      x = x[I]
      y = y[I]

      # Make curve monotonic
      if(self._monotonic):
         halfway = len(x) / 2
         for i in range(0, halfway):
            x[i] = np.min(x[i:halfway])
         for i in range(halfway, len(x)):
            x[i] = np.max(x[0:(i+1)])
      return x,y
