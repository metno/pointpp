import argparse
import numpy as np
import shutil
import sys
import matplotlib.pyplot as mpl
import netCDF4
import verif.input
import verif.metric
import verif.util
import pointpp.util
import pointpp.version
import pointpp.method
import pointpp.metric


def run(argv):
   methods = [x[0].lower() for x in pointpp.method.get_all()]
   parser = argparse.ArgumentParser(prog="pointpp", description="Point forecast post-processor")
   parser.add_argument('--version', action="version", version=pointpp.version.__version__)
   parser.add_argument('--debug', help="Show debug information", action="store_true")
   parser.add_argument('file', help="Verif NetCDF input file")
   parser.add_argument('-t', metavar="FILE", help="Verif NetCDF file to use for Training ", dest="file_training")
   parser.add_argument('-b', type=verif.util.parse_numbers, default=[100], metavar="BINS", help="Number of points in curve. If vector, then its the bin edges", dest="bins")
   parser.add_argument('-yb', type=verif.util.parse_numbers, default=[100], metavar="BINS", help="Number of points when using -y optimization. If vector, then its the bin edges", dest="ybins")
   parser.add_argument('-o', metavar="FILE", help="Output filename", dest="ofile")
   parser.add_argument('-m', metavar="METHOD", help="Optimization method.  Either a threshold-based score like ets, or one of: " + ', '.join(methods), required=True, dest="method")
   parser.add_argument('-loc', help="Post-process each station independently", dest="location_dependent", action="store_true")
   parser.add_argument('-lt', help="Post-process each leadtime independently", dest="leadtime_dependent", action="store_true")
   parser.add_argument('-tt', type=int, help="Training time, number of days to train the method", dest="ttime")
   parser.add_argument('-mono', default=False, help="Run monotonic filter on curve", action="store_true")
   parser.add_argument('-r', default=1, type=int, help="How many times to resample?", dest="resample")
   parser.add_argument('-mp', default=1, type=float, help="Use midpoint of range where score is above this percentage of the best", dest="midpoint")
   parser.add_argument('-mo', default=0, type=int, help="Minimum number of obs required to have a point in the curve", dest="min_obs")
   parser.add_argument('-ms', type=float, help="Minimum score to get a point", dest="min_score")
   parser.add_argument('-c', metavar="FILENAME", help="Write curve to this file", dest="curve_file")
   parser.add_argument('-cr', type=verif.util.parse_numbers, metavar="min,max", help="Min,max range for curve", dest="curve_minmax")
   parser.add_argument('-y', type=float, help="Create curve for this y value")
   parser.add_argument('-s', default="default", help="Solver to create curve, one of: fmin, sum, None", dest="solver")
   parser.add_argument('-d', type=parse_dates, help="Dates to do the training on", dest="dates")

   if len(sys.argv) == 1:
      parser.print_help()
      sys.exit(1)
   args = parser.parse_args()

   # Evaluation dataset
   input = verif.input.get_input(args.file)
   eobs = input.obs
   efcst = input.fcst

   # Training dataset
   if args.file_training is not None:
      input_training = verif.input.get_input(args.file_training)
   else:
      input_training = input
   tobs = input_training.obs
   tfcst = input_training.fcst

   if args.dates is not None:
       dates = [verif.util.unixtime_to_date(d) for d in input_training.times]
       Idates = [i for i in range(tobs.shape[0]) if dates[i] in args.dates]
       print "Reducing dates: %d to %d" % (tobs.shape[0], len(Idates))
       tobs = tobs[Idates, :, :]
       tfcst = tfcst[Idates, :, :]

   method = pointpp.method.get(args.method, args.bins, args.min_obs)
   if method is None:
      if args.method == "bias":
         metric = pointpp.metric.Bias()
      else:
         metric = verif.metric.get(args.method)
      if metric is not None:
         method = pointpp.method.MyMethod(metric, bins=args.bins,
               monotonic=args.mono, resample=args.resample,
               midpoint=args.midpoint, min_obs=args.min_obs,
               min_score=args.min_score, solver=args.solver)
         method._debug = args.debug
      else:
         method = None
      pointpp.util.DEBUG = args.debug

   tids = np.array([loc.id for loc in input_training.locations], int)
   eids = np.array([loc.id for loc in input.locations], int)

   """
   Not every location in the evaluation set is available in the training set.
   Find these locations and only calibrate when a location is available (if
   -loc is selected).
   """
   e2t_loc = dict()
   for id in eids:
      It = np.where(tids == id)[0]
      if len(It) == 1:
         Ie = np.where(eids == id)[0]
         num_valid_e = np.sum(np.isnan(efcst[:, :, Ie]*eobs[:, :, Ie]) == 0)
         num_valid_f = np.sum(np.isnan(tfcst[:, :, It]*tobs[:, :, It]) == 0)
         if num_valid_e > 10 and num_valid_f > 10:
            e2t_loc[Ie[0]] = It[0]
         else:
            pointpp.util.warning("Not enough valid data for location '%d'" % id)

   D = eobs.shape[0]
   LT = eobs.shape[1]
   LOC = eobs.shape[2]
   eobs2 = None

   if args.ofile is not None:
      """ Create output """
      shutil.copyfile(args.file, args.ofile)
      if args.method == "pers" or args.method == "fpers":
         """ Persistence methods should always be location and date dependent """
         efcst2 = np.nan * np.zeros(eobs.shape)
         for i in range(D):
            for j in e2t_loc:
               pointpp.util.progress_bar(i * LOC + j, D * LOC)
               jt = e2t_loc[j]
               efcst2 [i, :, j] = method.calibrate(tobs[i, :, jt], tfcst[i, :, jt], efcst[i, :, j])
      elif args.method == "clim":
         """ Climatology methods should always be location, leadtime, and month dependent """
         efcst2 = np.nan * np.zeros(eobs.shape)
         all_months = np.array([verif.util.unixtime_to_date(t) / 100 % 100 for t in input.times])
         months = np.unique(np.sort([verif.util.unixtime_to_date(t) / 100 % 100 for t in input.times]))
         for i in range(len(months)):
            month = months[i]
            I = np.where(all_months == month)[0]
            for i in range(LT):
               for j in e2t_loc:
                  jt = e2t_loc[j]
                  tmp = method.calibrate(tobs[I, i, jt].flatten(), tfcst[I, i, jt].flatten(), efcst[I, i, j].flatten())
                  efcst2[I, i, j] = tmp
      elif args.method == "anomaly":
         efcst2 = np.nan * np.zeros(eobs.shape)
         eobs2 = np.nan * np.zeros(eobs.shape)
         all_months = np.array([verif.util.unixtime_to_date(t) / 100 % 100 for t in input.times])
         months = np.unique(np.sort([verif.util.unixtime_to_date(t) / 100 % 100 for t in input.times]))
         for i in range(len(months)):
            month = months[i]
            I = np.where(all_months == month)[0]
            for j in e2t_loc:
               jt = e2t_loc[j]
               mean_obs = np.nanmean(tobs[I, :, jt])
               mean_fcst = np.nanmean(tfcst[I, :, jt])
               print j, jt, mean_obs, mean_fcst
               efcst2[I, :, j] = np.reshape(efcst[I, :, j] - mean_obs, [len(I), LT])
               eobs2[I, :, j] = np.reshape(eobs[I, :, j] - mean_obs, [len(I), LT])
      else:
         if not args.location_dependent and not args.leadtime_dependent:
            """ One big calibration """
            efcst2 = method.calibrate(tobs.flatten(), tfcst.flatten(), efcst.flatten())
            efcst2 = np.reshape(efcst2, efcst.shape)
         elif not args.location_dependent and args.leadtime_dependent:
            """ Separate calibration for each leadtime """
            efcst2 = np.nan * np.zeros(eobs.shape)
            for i in range(LT):
               tmp = method.calibrate(tobs[:, i, :].flatten(), tfcst[:, i, :].flatten(), efcst[:, i, :].flatten())
               efcst2[:, i, :] = np.reshape(tmp, [D, LOC])
         elif args.location_dependent and not args.leadtime_dependent:
            """ Separate calibration for each location """
            efcst2 = np.nan * np.zeros(eobs.shape)
            for j in e2t_loc:
               jt = e2t_loc[j]
               tmp = method.calibrate(tobs[:, :, jt].flatten(), tfcst[:, :, jt].flatten(), efcst[:, :, j].flatten())
               efcst2[:, :, j] = np.reshape(tmp, [D, LT])
         else:
            """ Separate calibration for each leadtime and location """
            efcst2 = np.nan * np.zeros(eobs.shape)
            for i in range(LT):
               for j in e2t_loc:
                  jt = e2t_loc[j]
                  efcst2[:, i, j] = method.calibrate(tobs[:, i, jt], tfcst[:, i, jt], efcst[:, i, j])

      fid = netCDF4.Dataset(args.ofile, 'a')
      print "Writing"
      fid.variables["fcst"][:] = efcst2
      if eobs2 is not None:
         fid.variables["obs"][:] = eobs2
      fid.close()

   if args.curve_file is not None:
      """ Create calibration curve """
      obs = tobs.flatten()
      fcst = tfcst.flatten()
      I = np.where((np.isnan(obs) == 0) & (np.isnan(fcst) == 0))[0]
      obs = obs[I]
      fcst = fcst[I]

      if args.y is not None:
         x, y, = method.get_single_curve(obs, fcst, args.y, args.ybins)
         if args.curve_file is not None:
            write(x, y, args.curve_file, "x score")
         else:
            mpl.plot(x, y, 'k-o')
            #q = [np.min(x), np.max(x)]
            #mpl.plot(q, q, '-', color="gray", lw=2)
            #mpl.gca().set_aspect(1)
            mpl.grid()
            mpl.show()
      else:
          cmin = np.min(fcst)
          cmax = np.max(fcst)
          if args.curve_minmax is not None:
              cmin = args.curve_minmax[0]
              cmax = args.curve_minmax[1]
          x, y, = method.get_curve(obs, fcst, cmin, cmax)
          if args.curve_file is not None:
             write(y, x, args.curve_file, "obs fcst")
          else:
             # x, y, lower, upper = method.get_curve(obs, fcst, np.min(fcst), np.max(fcst))
             for i in range(len(x)):
                print x[i], y[i]
             # import sys
             # sys.exit()
             #mpl.plot(fcst, obs, 'r.', alpha=0.3)
             mpl.plot(x, y, 'k-o')
             # mpl.plot(lower, y, 'k--o')
             # mpl.plot(upper, y, 'k--o')
             # mpl.plot((lower+upper)/2, y, 'k--o')
             mpl.plot([-15, 15], [-15, 15], '-', lw=2, color="gray")
             mpl.grid()
             mpl.gca().set_aspect(1)
             #mpl.xlim([-15, 15])
             #mpl.ylim([-15, 15])
             mpl.show()


def parse_dates(string):
    return verif.util.parse_numbers(string, True)


def write(x, y, filename, header=None):
   """
   Write data to file
   """
   if filename is not None:
      file = open(filename, 'w')
      if header is not None:
         file.write("%s\n" % header)
      for i in range(len(x)):
         file.write("%f %f\n" % (x[i], y[i]))
      file.close()


if __name__ == '__main__':
   run(sys.argv)
