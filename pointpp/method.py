import numpy as np
import matplotlib.pylab as mpl
import sys
import inspect
import verif.util
import verif.metric
import verif.interval
import pointpp.util
import time as timing
import sklearn.linear_model
import copy
import gridpp


def get_all():
    """
    Returns a dictionary of all classes where the key is the class
    name (string) and the value is the class object
    """
    temp = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    temp = [x for x in temp if x[0] not in ["Method", "Radpro"]]
    return temp


def get(name, bins, min_obs):
    """ Returns an instance of an object with the given class name """
    methods = get_all()
    m = None
    for method in methods:
        if name == method[0].lower():  # and method[1].is_valid():
            m = method[1](bins, min_obs)

    return m


def get_gridpp_metric(metric):
    if metric.get_class_name() == "Ets":
        name = gridpp.Ets
    elif metric.get_class_name() == "Kss":
        name = gridpp.Kss
    elif metric.get_class_name() == "Pc":
        name = gridpp.Pc
    elif metric.get_class_name() == "Bias":
        name = gridpp.Bias
    else:
        raise NotImplementedError
    return name


class Method(object):
    """
    Public interface
    """
    _debug = False
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

    def debug(self, msg):
        if self._debug:
            print(msg)


class Raw(Method):
    name = "Raw"

    def __init__(self, bins=None, min_obs=None):
        pass

    def _calibrate(self, Otrain, Ftrain, Feval):
        return Feval

    def get_curve(self, Otrain, Ftrain, xmin, xmax):
        x = np.linspace(xmin, xmax, 2)
        return [x, x]


class Clim(Method):
    name = "Climatology"

    def __init__(self, bins=None, min_obs=None):
        pass

    def calibrate(self, Otrain, Ftrain, Feval):
        mean = np.nanmean(Otrain)
        return mean * np.ones(len(Feval), 'float')

    def get_curve(self, Otrain, Ftrain, xmin, xmax):
        mean = np.nanmean(Otrain)
        x = np.linspace(xmin, xmax, self._num_bins)
        return [x,mean + 0*x]


class Pers(Method):
    name = "Persistence"

    def __init__(self, bins=None, min_obs=None):
        pass

    def calibrate(self, Otrain, Ftrain, Feval):
        """
        Don't use _calibrate here, since we don't want missing values to be
        removed, since then the first observation in the array isn't necessarily
        the right one
        """
        return Otrain[0] * np.ones(len(Feval))


class Fpers(Method):
    name = "Forecast persistence"

    def __init__(self, bins=None, min_obs=None):
        pass

    def _calibrate(self, Otrain, Ftrain, Feval):
        return Ftrain[0] * np.ones(len(Feval))


class Regression(Method):
    name = "Linear regression"

    def __init__(self, bins=None, min_obs=None):
        pass

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
        # x = np.linspace(xmin,xmax,self._num_bins)
        x = np.array([xmin, xmax])
        return [x,a+b*x]


class Multiplicative(Method):
    name = "Multiplicative correction"

    def __init__(self, bins=None, min_obs=None):
        pass

    def _calibrate(self, Otrain, Ftrain, Feval):
        b = self._get_coefficients(Otrain, Ftrain)
        return b * Feval

    def _get_coefficients(self, Otrain, Ftrain):
        mF = np.mean(Ftrain)
        mO = np.mean(Otrain)
        mOF = np.mean(Otrain*Ftrain)
        mFF = np.mean(Ftrain*Ftrain)
        if(mF != 0):
            b = (mO / mF)
        else:
            Common.warning("Unstable regression")
            b = 1
        return b


    def get_curve(self, Otrain, Ftrain, xmin, xmax):
        b = self._get_coefficients(Otrain, Ftrain)
        x = np.linspace(xmin,xmax,self._num_bins)
        return x, b*x


class Additive(Method):
    name = "Additive correction"

    def __init__(self, bins=None, min_obs=None):
        pass

    def _calibrate(self, Otrain, Ftrain, Feval):
        a = self._get_coefficients(Otrain, Ftrain)
        return a + Feval

    def _get_coefficients(self, Otrain, Ftrain):
        mF = np.mean(Ftrain)
        mO = np.mean(Otrain)
        a = (mO - mF)
        return a


    def get_curve(self, Otrain, Ftrain, xmin, xmax):
        a = self._get_coefficients(Otrain, Ftrain)
        x = np.linspace(xmin,xmax,self._num_bins)
        return x, a + x


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

        I0 = np.where(Feval < xmin)[0] # Below training set
        I1 = np.where(Feval > xmax)[0] # Above training set
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

        I = np.where((Feval <= xmax) & (Feval >= xmin))[0] # Inside training set
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

    def __init__(self, bins=None, min_obs=None):
        """ Resample curve to have this many points. Use None to use all datapoints """
        if bins is None:
            self._num_bins = 10

        else:
            if isinstance(bins, int):
                bins = [bins]
            self._num_bins = bins[0]

    def get_curve(self, Otrain, Ftrain, xmin, xmax):
        Fsort = np.sort(Ftrain)
        Osort = np.sort(Otrain)
        if xmin is not None and xmax is not None:
            I = np.where((Fsort >= xmin) & (Fsort <= xmax))[0]
            Fsort = Fsort[I]
            Osort = Osort[I]

        # Resample curve
        if(self._num_bins != None):
            x = np.linspace(min(Fsort), max(Fsort), self._num_bins)
            y = np.interp(x, Fsort, Osort)
        else:
            x = Fsort
            y = Osort

        return x, y


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

    def __init__(self, bins=[30], min_obs=0):
        if isinstance(bins, int):
            bins = [bins]
        self.bins = bins
        self.min_obs = min_obs

    def get_curve(self, Otrain, Ftrain, xmin, xmax):
        if len(self.bins) == 1:
            edges = np.linspace(xmin, xmax, self.bins[0] + 1)
        else:
            edges = self.bins
        x = np.zeros(len(edges)-1, 'float')
        y = np.copy(x)
        for i in range(0, len(x)):
            I = np.where((Ftrain >= edges[i]) & (Ftrain <= edges[i+1]) & (np.isnan(Ftrain) == 0) &
                  (np.isnan(Otrain) == 0))[0]
            if(len(I) > self.min_obs):
                x[i] = np.mean(Ftrain[I])
                y[i] = np.mean(Otrain[I])
                print(x[i], y[i], np.nanmax(Otrain[I]))
            else:
                x[i] = np.nan# (edges[i] + edges[i+1])/2
                y[i] = np.nan# x[i]
        I = np.where(np.isnan(x) == 0)[0]
        x = x[I]
        y = y[I]
        return [x,y]


class InverseConditional(Curve):
    """
    For each observation threshold, calculate the average forecasted
    F -> E^-1[F|O]
    What does this method optimize?
    """
    def __init__(self, bins=[30], min_obs=None):
        if isinstance(bins, int):
            bins = [bins]
        self.bins = bins

    name = "Inverse conditional"

    def get_curve(self, Otrain, Ftrain, xmin, xmax):
        if len(self.bins) == 1:
            edges = np.linspace(xmin, xmax, self.bins[0] + 1)
        else:
            edges = self.bins
        x = np.zeros(len(edges)-1, 'float')
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

    def __init__(self, metric, bins=[30], monotonic=True, resample=1, midpoint=1, min_obs=0, min_score=None, solver="default"):
        if isinstance(bins, int):
            bins = [bins]
        self._metric = metric
        self._monotonic = monotonic
        self._resample = resample
        self._midpoint = midpoint
        self._min_obs = min_obs
        self._min_score = min_score
        self._solver = solver
        try:
            self._bins = [int(bins)]
        except Exception as e:
            self._bins = bins
        self._num_dx = 100
        self._min_num_data = self._min_obs

    @property
    def name(self):
        className = self._metric.get_class_name()
        if className in ["BiasFreq", "Bias"]:
            className = "Bias"
        else:
            className = className.upper()
        return className + "-optimizer"

    def compute_scores(self, Otrain, Ftrain, obs_threshold, fcst_thresholds):
        interval = verif.interval.Interval(obs_threshold, np.inf, False, True)
        scores = np.zeros(len(fcst_thresholds))
        for k in range(len(fcst_thresholds)):
            f_interval = verif.interval.Interval(fcst_thresholds[k], np.inf, False, True)
            scores[k] = self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, f_interval)
        if self._metric.orientation != 1:
            scores = -scores

        Ivalid = np.where(np.isnan(scores) == 0)[0]
        fcst_thresholds = fcst_thresholds[Ivalid]
        scores = scores[Ivalid]
        return scores, fcst_thresholds

    def get_starting_point(self, Otrain, Ftrain, y):
        """
        Find a good estimate for x given a y. Compute the score for the range of
        observations and pick the x value giving the best score
        """
        return self.get_curve_fmin(Otrain, Ftrain, [y])
        # xx = np.linspace(np.min(Otrain), np.max(Otrain), 50)
        # scores, xx = self.compute_scores(Otrain, Ftrain, y, xx)
        # bestScore = np.max(scores)
        # Ibest = np.where(scores == bestScore)[0]
        # x = xx[Ibest[0]]
        return x

    def get_single_curve(self, Otrain, Ftrain, threshold, bins=[100]):
        if len(bins) == 1:
            xmin = np.nanmin(Ftrain)
            xmax = np.nanmax(Ftrain)
            x = np.linspace(xmin, xmax, bins[0])
        else:
            x = np.array(bins)

        scores = np.zeros(len(x))
        metric = get_gridpp_metric(self._metric)
        for k in range(0,len(x)):
            interval = verif.interval.Interval(threshold, np.inf, False, True)
            f_interval = verif.interval.Interval(x[k], np.inf, False, True)
            if self._resample == 1:
                scores[k] = gridpp.calc_score(Otrain, Ftrain, threshold, x[k], metric)
            else:
                raise NotImplementedError
                # scores[k] = self._metric.compute_from_obs_fcst_resample(Otrain, Ftrain, self._resample, interval, f_interval)
        # if self._metric.orientation != 1:
        #   scores = -scores
        return x, scores

    def get_curve(self, Otrain, Ftrain, xmin, xmax):
        if len(self._bins) > 1:
            thresholds = self._bins
        else:
            # thresholds = gridpp.calc_even_quantiles(np.sort(Otrain), int(self._bins[0]))
            thresholds = np.linspace(xmin, xmax, self._bins[0])
        metric = get_gridpp_metric(self._metric)
        c0, c1 = gridpp.metric_optimizer_curve(Otrain, Ftrain, thresholds, metric)
        if self._monotonic:
            c0, c1 = gridpp.monotonize_curve(c0, c1)
        return c0, c1

    def get_curve_orig(self, Otrain, Ftrain, xmin, xmax):
        if len(self._bins) == 1:
            y = np.linspace(xmin, xmax, self._bins[0])
        else:
            y = np.copy(self._bins)

        I = np.where((y >= np.min(Otrain)) & (y <= np.max(Otrain)))[0]
        assert(len(I) > 0)
        y = y[I]

        if(0 and self._min_obs > 0):
            sortobs = np.sort(Otrain)
            if len(sortobs) < self._min_obs*2:
                verif.util.error("Too few data points when min_obs is set")
            I = np.where((y >= sortobs[self._min_obs]) & (y <= sortobs[-self._min_obs]))[0]
            y = y[I]
        # y = np.array([-15, -14, -13, -10, -5, 0, 5])

        import time
        s = time.time()
        if self._solver == "default":
            x = self.get_curve_default(Otrain, Ftrain, y)
        elif self._solver == "new":
            x = self.get_curve_new(Otrain, Ftrain, y)
        elif self._solver == "fmin":
            x = self.get_curve_fmin(Otrain, Ftrain, y)
        elif self._solver == "fminnew":
            x = self.get_curve_fmin_new(Otrain, Ftrain, y)
        elif self._solver == "sum":
            x = self.get_curve_sum(Otrain, Ftrain, y)
        elif self._solver == "old":
            x = self.get_curve_old(Otrain, Ftrain, y)
        else:
            verif.util.error("Invalid solver")
        # sys.exit()

        if self._monotonic:
            halfway = len(x) // 2
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

        # Remove points where there are too few obs/fcst
        if self._min_num_data > 0:
            obs_sort = np.sort(Otrain)
            fcst_sort = np.sort(Ftrain)
            assert(len(obs_sort) > 2*self._min_num_data)
            I = np.where((x >= fcst_sort[self._min_num_data]) & (x <= fcst_sort[-self._min_num_data]) &\
                         (y >= obs_sort[self._min_num_data]) & (y <= obs_sort[-self._min_num_data]))[0]
            print("Removing %d of %d" % (len(x) - len(I), len(x)))
            x = x[I]
            y = y[I]

        """
        for i in range(len(x)):
           print("%2.3f %2.3f" % (x[i], y[i]))
        print("%2.3f %2.3f" % (np.std(x), np.std(y)))
        print("%2.3f s" % (time.time() - s))
        import sys
        sys.exit()
        """

        return x, y

    def get_curve_fmin(self, Otrain, Ftrain, y):
        """
        Pros: Fast
        Cons: Not robust when the peak is flat
        """
        x = np.nan*np.ones(len(y), 'float')
        import scipy.optimize

        for i in range(0, len(y)):
            interval = verif.interval.Interval(y[i], np.inf, False, True)
            sign = - self._metric.orientation
            f = lambda x: sign * self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, verif.interval.Interval(x, np.inf, False, True))
            x[i] = scipy.optimize.fmin(f, y[i], xtol=0.1, disp=False)

        return x

    def get_curve_fmin_new(self, Otrain, Ftrain, y):
        """
        Pros: Fast
        Cons: Not robust when the peak is flat
        """
        x = np.nan*np.ones(len(y), 'float')
        import scipy.optimize
        upper_half = range(len(y)/2, len(y))
        lower_half = range(0, len(y)/2)[::-1]

        for yrange in [upper_half, lower_half]:
            lastDiff = 0
            for i in yrange:
                interval = verif.interval.Interval(y[i], np.inf, False, True)
                self.debug("%f Searching (%f %f)" % (y[i], -np.inf, np.inf))
                f = lambda x: -self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, verif.interval.Interval(x, np.inf, False, True))
                x[i] = scipy.optimize.fmin(f, y[i] - lastDiff, xtol=0.1, disp=False)
                lastDiff = y[i] - x[i]

        return x

    def get_curve_sum(self, Otrain, Ftrain, y):
        """
        Pros: Robust
        Cons: Does it work for PC?
        """
        x = np.nan*np.ones(len(y), 'float')
        scores = np.zeros([len(y)], 'float')
        # Loop over observation thresholds
        for i in range(0, len(y)):
            # Integrate over all forecast thresholds
            for k in range(0,len(y)):
                interval = verif.interval.Interval(y[i], np.inf, False, True)
                f_interval = verif.interval.Interval(y[k], np.inf, False, True)
                scores[k] = self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, f_interval)

            Ivalid = np.where(np.isnan(scores) == 0)[0]
            total_score = np.nansum(scores)
            if not np.isnan(total_score) and total_score != 0:
                x[i] = np.nansum(scores * y) / total_score

        return x

    def get_curve_new(self, Otrain, Ftrain, y):
        x = np.nan*np.ones(len(y), 'float')
        lastX  = np.nanmean(Otrain)

        middleX = self.get_starting_point(Otrain, Ftrain, y[len(y)/2])
        # middleX = self.get_starting_point(Otrain, Ftrain, y[0])
        """
        Start creating the line from the middle. That is process the upper half
        first, starting from the bottom, then process the lower half from the top.
        """
        upper_half = range(len(y)/2, len(y))
        lower_half = range(0, len(y)/2)[::-1]

        for yrange in [upper_half, lower_half]:
            # Reset
            hit_edge = False
            lastX = middleX

            for i in yrange:
                """ Skip the rest of the line if we have hit the edges """
                if hit_edge:
                    continue

                xx = np.linspace(lastX - 10, lastX + 10, self._num_dx)
                self.debug("%f Searching (%f %f)" % (y[i], xx[0], xx[-1]))

                scores, xx = self.compute_scores(Otrain, Ftrain, y[i], xx)

                if len(scores) > 0:
                    # Find the best score
                    bestScore = np.max(scores)
                    Ibest = np.where(scores == bestScore)[0]
                    if self._midpoint == 1:
                        if len(Ibest) > 1:
                            # Multiple best ones
                            II = np.where(xx[Ibest] > lastX)[0]
                            if len(II) > 0:
                                # Use the nearest threshold above the previous
                                x[i] = xx[Ibest[II[0]]]
                            else:
                                # Use the highest possible threshold
                                # x[i] = y[i]
                                # The following
                                x[i] = xx[Ibest[-1]]
                        else:
                            if(Ibest == len(scores)):
                                Common.error("Edge problem")
                            x[i] = xx[Ibest]
                    else:
                        ref = np.max(scores[0], scores[-1])
                        Ilower = np.where((scores-ref) > self._midpoint * (bestScore-ref))[0]
                        if len(Ilower) > 0:
                            lower = xx[Ilower[0]]
                            upper = xx[Ilower[-1]]
                            midpoint = (lower + upper)/2
                            x[i] = midpoint

                    # Don't make a point if the score is too low
                    if self._metric.orientation == 1 and self._min_score is not None and bestScore < self._min_score:
                        self.debug("Removing")
                        x[i] = np.nan
                    elif np.max(scores) - np.min(scores) < 0:
                        self.debug("Not enough spread in scores")
                        x[i] = np.nan
                    # If the score at the edge is best, set to extreme most forecast
                    elif(scores[0] > bestScore*0.9999):
                        self.debug("Lower edge")
                        hit_edge = True
                        x[i] = np.nanmin(Ftrain)
                    elif(scores[-1] > bestScore*0.9999):
                        self.debug("Upper edge")
                        hit_edge = True
                        x[i] = np.nanmax(Ftrain)
                        x[i] = xx[-1]
                # No valid data, use monotonic
                else:
                    self.debug("No valid data for %f" % y[i])
                    if(i > 1):
                        x[i] = x[i-1]
                    else:
                        x[i] = y[i]
                    dx = 0

                if not np.isnan(x[i]):
                    lastX = x[i]
                    self.debug("LastX %f" % (lastX))

        return x

    def get_curve_default(self, Otrain, Ftrain, y):
        x = np.nan*np.ones(len(y), 'float')
        scores = np.zeros([self._num_dx], 'float')
        lastX  = np.min(y)-8
        started = False
        for i in range(0, len(y)):
            # pointpp.util.progress_bar(i, len(y))
            """
            Compute the score for each possible perturbation
            """
            # Determine dxs
            if not started:
                # On the first iteration, test a large range of values
                xx = y[0] + np.linspace(-30, 30, self._num_dx) # 0.1 increments
                started = True
            else:
                # The change in dx from one threshold to the next is unlikely to change a lot
                xx = np.linspace(lastX - 8, lastX + 8, self._num_dx)
            self.debug("%f Searching (%f %f)" % (y[i], xx[0], xx[-1]))

            """
            Compute the score for each possible perturbation
            """
            for k in range(len(xx)):
                interval = verif.interval.Interval(y[i], np.inf, False, True)
                f_interval = verif.interval.Interval(xx[k], np.inf, False, True)
                scores[k] = self._metric.compute_from_obs_fcst(Otrain, Ftrain, interval, f_interval)
            if self._metric.orientation != 1:
                scores = -scores

            Ivalid = np.where(np.isnan(scores) == 0)[0]
            xx = xx[Ivalid]
            scores = scores[Ivalid]
            if len(Ivalid) > 0:
                # Find the best score
                bestScore = np.max(scores)
                Ibest = np.where(scores == bestScore)[0]
                if self._midpoint == 1:
                    if len(Ibest) > 1:
                        # Multiple best ones
                        II = np.where(xx[Ibest] > lastX)[0]
                        if len(II) > 0:
                            # Use the nearest threshold above the previous
                            x[i] = xx[Ibest[II[0]]]
                        else:
                            # Use the highest possible threshold
                            # x[i] = y[i]
                            # The following
                            x[i] = xx[Ibest[-1]]
                    else:
                        if(Ibest == len(scores)):
                            Common.error("Edge problem")
                        x[i] = xx[Ibest]
                else:
                    ref = np.max(scores[0], scores[-1])
                    Ilower = np.where((scores-ref) > self._midpoint * (bestScore-ref))[0]
                    if len(Ilower) > 0:
                        lower = xx[Ilower[0]]
                        upper = xx[Ilower[-1]]
                        midpoint = (lower + upper)/2
                        x[i] = midpoint

                if 1:
                    # Don't make a point if the score is too low
                    if self._metric.orientation == 1 and self._min_score is not None and bestScore < self._min_score:
                        self.debug("Removing")
                        x[i] = np.nan
                    elif np.max(scores) - np.min(scores) < 0:
                        self.debug("Not enough spread in scores")
                        x[i] = np.nan
                    # If the score at the edge is best, set to extreme most forecast
                    elif(scores[0] > bestScore*0.999):
                        self.debug("Lower edge")
                        x[i] = np.nanmin(Ftrain)
                    elif(scores[-1] > bestScore*0.999):
                        self.debug("Upper edge")
                        x[i] = np.nanmax(Ftrain)
            # No valid data, use monotonic
            else:
                self.debug("No valid data for %f" % y[i])
                if(i > 1):
                    x[i] = x[i-1]
                else:
                    x[i] = y[i]
                dx = 0
            if not np.isnan(x[i]):
                lastX = x[i]
                self.debug("LastX %f" % (lastX))

        if 1:
            # Remove repeated end points
            I = np.where(x == np.min(Ftrain))[0]
            if(len(I) > 1):
                x[I[0:-1]] = np.nan
            I = np.where(x == np.max(Ftrain))[0]
            if(len(I) > 1):
                x[I[1:]] = np.nan

        return x

    def get_curve_old(self, Otrain, Ftrain, y):
        x = np.nan*np.ones(len(y), 'float')
        num_bins = len(self._bins) if len(self._bins) > 1 else self._bins[0]
        scores = np.zeros([num_bins], 'float')
        lastX  = np.min(y)-8
        lastDx = None
        for i in range(0, len(y)):
            """
            Compute the score for each possible perturbation
            """
            # Determine dxs
            if(lastDx == None):
                # On the first iteration, test a large range of values
                dxs = np.linspace(-30,30,self._num_dx) # 0.1 increments
            else:
                # The change in dx from one threshold to the next is unlikely to change a lot
                # This gives slighly different results than the default method,
                # mainly due to the exact numerics of this:
                dxs = np.linspace(lastDx - 8,lastDx + 8, self._num_dx)

            currY = y[i]
            self.debug("%f Searching (%f %f)" % (y[i], currY-dxs[-1], currY-dxs[0]))

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
                        self.debug("Removing")
                        x[i] = np.nan
                    elif np.max(scores) - np.min(scores) < 0:
                        self.debug("Not enough spread in scores")
                        x[i] = np.nan
                    # If the score at the edge is best, set to extreme most forecast
                    elif(scores[-1] > bestScore*0.999):
                        self.debug("Upper edge")
                        x[i] = np.nanmin(Ftrain)
                    elif(scores[0] > bestScore*0.999):
                        self.debug("Lower edge")
                        x[i] = np.nanmax(Ftrain)
            # No valid data, use monotonic
            else:
                self.debug("No valid data for %f" % y[i])
                if(i > 1):
                    x[i] = x[i-1]
                else:
                    x[i] = currY
                dx = 0
            lastX = x[i]
            lastDx = dx
            self.debug("LastX %f %f" % (lastX, lastDx))

        if 1:
            # Remove repeated end points
            I = np.where(x == np.min(Ftrain))[0]
            if(len(I) > 1):
                x[I[0:-1]] = np.nan
            I = np.where(x == np.max(Ftrain))[0]
            if(len(I) > 1):
                x[I[1:]] = np.nan

        return x


class Radpro(object):
    def __init__(self):
        self.turn_off_with_precip = False
        self.weight_recent = None
        self.weight_yesterday = None

    def train(self, obs, fcst):
        """
        Train the internal parameters on a training dataset

        Arguments:
           obs (np.array):

        Returns:
           np.array: Weights of recent bias
           np.array: Weights of yesterday's bias
        """
        D = obs.shape[0]
        LT = obs.shape[1]
        L = obs.shape[2]
        w1 = np.zeros(LT)
        w2 = np.zeros(LT)
        l = 0
        o = obs[:, :, :]
        f = fcst[:, :, :]
        bias = o - f

        model = sklearn.linear_model.LinearRegression()

        for lt in range(LT):
            day = int(np.floor(lt / 24.0)) + 1
            p = np.zeros([(D-day)*L, 2])
            a = np.zeros([(D-day)*L])
            for l in range(L):
                I = range(l*(D-day), (l+1)*(D-day))
                p[I, 0] = bias[day:, 0, l]
                p[I, 1] = bias[0:-day, lt, l]
                a[I] = bias[day:, lt, l]
            I = np.where(np.isnan(np.min(p, axis=1) + a)==0)[0]
            model.fit(p[I, :], a[I])
            w1[lt] = model.coef_[0]
            w2[lt] = model.coef_[1]

        self.weight_recent = w1
        self.weight_yesterday = w2
        return w1, w2

    def apply(self, obs, fcst):
        """ Apply the parameters to an evaluation dataset """
        D = obs.shape[0]
        LT = obs.shape[1]
        L = obs.shape[2]
        efcst = copy.deepcopy(fcst)
        for lt in range(LT):
            day = int(np.floor(lt / 24.0)) + 1
            for d in range(day, D):
                recent = obs[d, 0, :] - fcst[d, 0, :]
                yesterday = obs[d - day, lt, :] - fcst[d - day, lt, :]
                efcst[d, lt, :] = efcst[d, lt, :] + self.weight_recent[lt] * recent
                + self.weight_yesterday[lt] * yesterday

        return efcst
