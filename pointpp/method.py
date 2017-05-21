import numpy as np
import matplotlib.pylab as mpl
import sys
import inspect
import verif.util
import verif.metric
import verif.interval
import pointpp.util


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
      if len(I) == 0:
         return np.nan*np.zeros(Feval.shape)
      Ieval = np.where(np.isnan(Feval) == 0)[0]
      x = np.nan*np.zeros(Feval.shape)
      if len(Ieval) > 0:
         x[Ieval] = self._calibrate(Otrain[I], Ftrain[I], Feval[Ieval])
      return x

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


class Pers(Method):
   name = "Persistence"

   def __init__(self, nbins=None):
      pass

   def _calibrate(self, Otrain, Ftrain, Feval):
      return Otrain[0] * np.ones(len(Feval))


class Fpers(Method):
   name = "Forecast persistence"

   def __init__(self, nbins=None):
      pass

   def _calibrate(self, Otrain, Ftrain, Feval):
      return Ftrain[0] * np.ones(len(Feval))


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


class Qq(Curve):
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

   def __init__(self, metric, nbins=30, monotonic=True, resample=1, midpoint=1, min_obs=0, min_score=None):
      self._metric = metric
      self._nbins  = nbins
      self._monotonic = monotonic
      self._resample = resample
      self._midpoint = midpoint
      self._min_obs = min_obs
      self._min_score = min_score

   def name(self):
      className = self._metric.getClassName()
      if className == "BiasFreq":
         className = "Bias"
      else:
         className = className.upper() 
      return className + "-optimizer"

   def get_single_curve(self, Otrain, Ftrain, threshold, xmin=None, xmax=None):
      if xmin is None:
         xmin = np.nanmin(Ftrain)
      if xmax is None:
         xmax = np.nanmax(Ftrain)
      x = np.linspace(xmin, xmax, 100)
      scores = np.zeros(len(x))
      for k in range(0,len(x)):
         interval = verif.interval.Interval(threshold, np.inf, False, True)
         f_interval = verif.interval.Interval(x[k], np.inf, False, True)
         if self._resample == 1:
            scores[k] = self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, f_interval)
         else:
            scores[k] = self._metric.compute_from_obs_fcst_resample(Otrain, Ftrain, self._resample, interval, f_interval)
      if self._metric.orientation != 1:
         scores = -scores
      return x, scores

   def get_curve_fmin(self, Otrain, Ftrain, xmin, xmax):
      y = np.linspace(xmin, xmax, self._nbins)
      I = np.where((y >= np.min(Otrain)) & (y <= np.max(Otrain)))[0]
      assert(len(I) > 0)
      y = y[I]

      if(self._min_obs > 0):
         sortobs = np.sort(Otrain)
         if len(sortobs) < self._min_obs*2:
            print "Too few data points when min_obs is set"
            sys.exit()
         I = np.where((y >= sortobs[self._min_obs]) & (y <= sortobs[-self._min_obs]))[0]
         y = y[I]

      x = np.nan*np.ones(len(y), 'float')
      import scipy.optimize

      for i in range(0, len(y)):
         # pointpp.util.progress_bar(float(i) / len(y), 80) 
         interval = verif.interval.Interval(y[i], np.inf, False, True)
         f = lambda x: -self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, verif.interval.Interval(x, np.inf, False, True))
         f2 = lambda x: x*self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, verif.interval.Interval(x, np.inf, False, True))
         fs = lambda x: self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, verif.interval.Interval(x, np.inf, False, True))
         max_x = scipy.optimize.fmin(f, y[i], xtol=0.1, disp=False)
         x[i] = max_x

      return x, y

   def get_curve_sum(self, Otrain, Ftrain, xmin, xmax):
      y = np.linspace(xmin, xmax, self._nbins)
      I = np.where((y >= np.min(Otrain)) & (y <= np.max(Otrain)))[0]
      assert(len(I) > 0)
      y = y[I]

      if(self._min_obs > 0):
         sortobs = np.sort(Otrain)
         if len(sortobs) < self._min_obs*2:
            print "Too few data points when min_obs is set"
            sys.exit()
         I = np.where((y >= sortobs[self._min_obs]) & (y <= sortobs[-self._min_obs]))[0]
         y = y[I]

      x = np.nan*np.ones(len(y), 'float')
      scores = np.zeros([len(y)], 'float')
      for i in range(0, len(y)):
         """
         Compute the score for each possible perturbation
         """
         for k in range(0,len(y)):
            interval = verif.interval.Interval(y[i], np.inf, False, True)
            f_interval = verif.interval.Interval(y[k], np.inf, False, True)
            scores[k] = self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, f_interval)
         if self._metric.orientation != 1:
            scores = -scores

         Ivalid = np.where(np.isnan(scores) == 0)[0]
         if len(Ivalid) > 0:
            x[i] = np.sum(scores[Ivalid] * y[Ivalid]) / np.sum(scores[Ivalid])

      # Make curve monotonic
      if(self._monotonic):
         halfway = len(x) / 2
         for i in range(0, halfway-1):
            if x[i] > np.nanmin(x[i:-1]):
               x[i] = np.nan
            # x[i] = np.min(x[i:halfway])
         for i in range(halfway+1, len(x)):
            if x[i] < np.nanmax(x[0:i]):
               x[i] = np.nan
            # x[i] = np.max(x[0:(i+1)])

      # Remove missing
      I = np.where((np.isnan(x) == 0) & (np.isnan(y) == 0))[0]
      x = x[I]
      y = y[I]

      return x, y

   def get_curve(self, Otrain, Ftrain, xmin, xmax):
      y = np.linspace(xmin, xmax, self._nbins)
      I = np.where((y >= np.min(Otrain)) & (y <= np.max(Otrain)))[0]
      assert(len(I) > 0)
      y = y[I]

      if(self._min_obs > 0):
         sortobs = np.sort(Otrain)
         if len(sortobs) < self._min_obs*2:
            print "Too few data points when min_obs is set"
            sys.exit()
         I = np.where((y >= sortobs[self._min_obs]) & (y <= sortobs[-self._min_obs]))[0]
         y = y[I]

      x = np.nan*np.ones(len(y), 'float')
      lower = -np.ones(len(y), 'float')
      upper = -np.ones(len(y), 'float')
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

         """
         Compute the score for each possible perturbation
         """
         for k in range(0,len(dxs)):
            dx = dxs[k]
            interval = verif.interval.Interval(currY, np.inf, False, True)
            f_interval = verif.interval.Interval(currY - dx, np.inf, False, True)
            if 0:
               #if self._resample == 1:
               temp = 0
               for t in range(self._resample):
                  II = np.random.randint(0, len(Otrain), len(Otrain)/2)
                  temp += self._metric.compute_from_obs_fcst(Otrain[II], Ftrain[II], interval, f_interval)
               scores[k] = temp / self._resample
            else:
               scores[k] = self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, f_interval)
               #else:
               #   scores[k] = self._metric.compute_from_obs_fcst_resample(Otrain, Ftrain, self._resample, interval, f_interval)
         if self._metric.orientation != 1:
            scores = -scores

         # Smooth the scores
         #scores = np.convolve(scores, [1,1,1,1,1,1,1], 'same')

         Ivalid = np.where(np.isnan(scores) == 0)[0]
         if len(Ivalid) > 0:
            # Find the best score
            bestScore = np.max(scores[Ivalid])
            Ibest = np.where(scores[Ivalid] == bestScore)[0]
            if len(Ibest) > 1:
               # Multiple best ones
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

            if self._midpoint != 1:
               Ilower = np.where(scores > self._midpoint * bestScore)[0]
               if len(Ilower) > 0:
                  lower = currY - dxs[Ilower[0]]
                  upper = currY - dxs[Ilower[-1]]
               midpoint = (lower + upper)/2
               x[i] = midpoint
               dx = currY - x[i]

            if 1:
               # Don't make a point if the score is too low
               if self._metric.orientation == 1 and self._min_score is not None and bestScore < self._min_score:
                  # print "Removing"
                  x[i] = np.nan
               elif np.max(scores) - np.min(scores) < 0:
                  # print "Not enough spread in scores"
                  x[i] = np.nan
               # If the score at the edge is best, set to extreme most forecast
               elif(scores[-1] > bestScore*0.999):
                  # print "Upper edge"
                  x[i] = np.nanmin(Ftrain)
               elif(scores[0] > bestScore*0.999):
                  # print "Lower edge"
                  x[i] = np.nanmax(Ftrain)
         # No valid data, use monotonic
         else:
            print "No valid data for %f" % y[i]
            if(i > 1):
               x[i] = x[i-1]
            else:
               x[i] = currY
            dx = 0
         lastX = x[i]
         lastDx = dx

      if 1:
         # Remove repeated end points
         I = np.where(x == np.min(Ftrain))[0]
         if(len(I) > 1):
            x[I[0:-1]] = np.nan
         I = np.where(x == np.max(Ftrain))[0]
         if(len(I) > 1):
            x[I[1:]] = np.nan

      # Make curve monotonic
      if(self._monotonic):
         halfway = len(x) / 2
         for i in range(0, halfway-1):
            if x[i] > np.nanmin(x[i:-1]):
               x[i] = np.nan
            # x[i] = np.min(x[i:halfway])
         for i in range(halfway+1, len(x)):
            if x[i] < np.nanmax(x[0:i]):
               x[i] = np.nan
            # x[i] = np.max(x[0:(i+1)])

      # Remove missing
      I = np.where((np.isnan(x) == 0) & (np.isnan(y) == 0))[0]
      x = x[I]
      y = y[I]

      return x, y
