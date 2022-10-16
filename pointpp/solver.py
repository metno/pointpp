class Solver(object):
    def get(self, Otrain, Ftrain, xmin, xmax):
        raise NotImplementedError()

class Fmin(Solver):
    def get(self, Otrain, Ftrain, xmin, xmax):
        y = np.linspace(xmin, xmax, self._nbins)
        I = np.where((y >= np.min(Otrain)) & (y <= np.max(Otrain)))[0]
        assert(len(I) > 0)
        y = y[I]

        if(self._min_obs > 0):
            sortobs = np.sort(Otrain)
            if len(sortobs) < self._min_obs*2:
                print("Too few data points when min_obs is set")
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
