import argparse
import numpy as np
import shutil
import verif.input
import verif.metric
import pointpp.util
import pointpp.version
import pointpp.method
import matplotlib.pyplot as mpl
import netCDF4


def run(argv):
   parser = argparse.ArgumentParser(prog="ppverif", description="Hybrid weather generator, combining stochastic and physical modelling")
   parser.add_argument('--version', action="version", version=pointpp.version.__version__)
   parser.add_argument('--debug', help="Show debug information", action="store_true")
   parser.add_argument('file', help="Input file")
   parser.add_argument('-t', help="Training file", dest="file_training")
   parser.add_argument('-b', type=int, default=100, metavar="NUM", help="Number of bins", dest="num_bins")
   parser.add_argument('-o', metavar="FILE", help="Output filename", dest="ofile")
   parser.add_argument('-m', metavar="METHOD", help="Optimization method", required=True, dest="method")
   parser.add_argument('-loc', help="Post-process each station independently?", dest="location_dependent", action="store_true")
   parser.add_argument('-lt', help="Post-process each leadtime independently?", dest="leadtime_dependent", action="store_true")
   parser.add_argument('-tt', type=int, help="Training time", dest="ttime")
   parser.add_argument('-e', help="Evaluation set", dest="efile")
   parser.add_argument('-mono', default=False, help="Make curve monotonic", action="store_true")
   parser.add_argument('-r', default=1, type=int, help="How many times to resample?", dest="resample")
   parser.add_argument('-mp', default=1, type=float, help="Use midpoint of range where score is above this percentage of the best", dest="midpoint")
   parser.add_argument('-mo', default=0, type=int, help="Minimum number of obs required to have a point in the curve", dest="min_obs")
   parser.add_argument('-ms', type=float, help="Minimum score to get a point", dest="min_score")
   parser.add_argument('-c', help="Write curve to this file", dest="curve_file")
   parser.add_argument('-y', type=float, help="Create curve for this y value")
   parser.add_argument('-s', default="default", help="One of fmin sum or None", dest="solver")

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

   method = pointpp.method.get(args.method)
   if method is None:
      metric = verif.metric.get(args.method)
      if metric is not None:
         method = pointpp.method.MyMethod(metric, nbins=args.num_bins,
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
         """ Climatology methods should always be location and month dependent """
         efcst2 = np.nan * np.zeros(eobs.shape)
         all_months = np.array([verif.util.unixtime_to_date(t) / 100 % 100 for t in input.times])
         months = np.unique(np.sort([verif.util.unixtime_to_date(t) / 100 % 100 for t in input.times]))
         for i in range(len(months)):
            month = months[i]
            I = np.where(all_months == month)[0]
            for j in e2t_loc:
               jt = e2t_loc[j]
               tmp = method.calibrate(tobs[I, :, jt].flatten(), tfcst[I, :, jt].flatten(), efcst[I, :, j].flatten())
               efcst2[I, :, j] = np.reshape(tmp, [len(I), LT])
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
   else:
      """ Create calibration curve """
      obs = tobs[:, -1, :].flatten()
      fcst = tfcst[:, -1, :].flatten()
      I = np.where((np.isnan(obs) == 0) & (np.isnan(fcst) == 0))[0]
      obs = obs[I]
      fcst = fcst[I]
      if args.y is not None:
         x, y, = method.get_single_curve(obs, fcst, args.y)
         mpl.plot(x, y, 'k-o')
         #q = [np.min(x), np.max(x)]
         #mpl.plot(q, q, '-', color="gray", lw=2)
         #mpl.gca().set_aspect(1)
         mpl.grid()
         mpl.show()
      else:
          import time as timing
          s = timing.time()
          x, y, = method.get_curve(obs, fcst, np.min(fcst), np.max(fcst))
          print timing.time() - s
          if args.curve_file is not None:
             file = open(args.curve_file, 'w')
             file.write("unixtime obs fcst\n")
             for i in range(len(x)):
                file.write("%d %f %f\n" % (1e9 + i,  x[i], y[i]))
             file.close()
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


if __name__ == '__main__':
   main()
