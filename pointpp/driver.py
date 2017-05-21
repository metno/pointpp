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

   input = verif.input.get_input(args.file)
   obs = input.obs.flatten()
   fcst = input.fcst.flatten()
   obs_ar = input.obs
   fcst_ar = input.fcst

   method = pointpp.method.get(args.method)
   if method is None:
      metric = verif.metric.get(args.method)
      if metric is None:
         verif.util.error("Could not understand '%s'" % args.method)

      method = pointpp.method.MyMethod(metric, nbins=args.num_bins,
            monotonic=args.mono, resample=args.resample,
            midpoint=args.midpoint, min_obs=args.min_obs,
            min_score=args.min_score, solver=args.solver)
      method._debug = args.debug

   D = obs_ar.shape[0]
   LT = obs_ar.shape[1]
   LOC= obs_ar.shape[2]

   if args.ofile is not None:
      """ Create output """
      shutil.copyfile(args.file, args.ofile)
      if args.method == "pers" or args.method == "fpers":
         """ Persistence methods should always be location and date independent """
         fcst2_ar = np.nan * np.zeros(obs_ar.shape)
         for i in range(D):
            for j in range(LOC):
               pointpp.util.progressbar(i * LOC + j, D * LOC)
               fcst2_ar [i, :, j] = method.calibrate(obs_ar[i, :, j], fcst_ar[i, :, j], fcst_ar[i, :, j])
      else:
         if not args.location_dependent and not args.leadtime_dependent:
            """ One big calibration """
            fcst2 = method.calibrate(obs, fcst, fcst)
            fcst2_ar = np.reshape(fcst2, fcst_ar.shape)
         elif not args.location_dependent and args.leadtime_dependent:
            """ Separate calibration for each leadtime """
            fcst2_ar = np.nan * np.zeros(obs_ar.shape)
            for i in range(LT):
               tmp = method.calibrate(obs_ar[:, i, :].flatten(), fcst_ar[:, i, :].flatten(), fcst_ar[:, i, :].flatten())
               fcst2_ar[:, i, :] = np.reshape(tmp, [D, LOC])
         elif args.location_dependent and not args.leadtime_dependent:
            """ Separate calibration for each location """
            fcst2_ar = np.nan * np.zeros(obs_ar.shape)
            for i in range(LOC):
               tmp = method.calibrate(obs_ar[:, :, i].flatten(), fcst_ar[:, :, i].flatten(), fcst_ar[:, :, i].flatten())
               fcst2_ar[:, :, i] = np.reshape(tmp, [D, LT])
         else:
            """ Separate calibration for each leadtime and location """
            fcst2_ar = np.nan * np.zeros(obs_ar.shape)
            for i in range(LT):
               for j in range(LOC):
                  fcst2_ar[:, i, j] = method.calibrate(obs_ar[:, i, j], fcst_ar[:, i, j], fcst_ar[:, i, j])

      fid = netCDF4.Dataset(args.ofile, 'a')
      print "Writing"
      fid.variables["fcst"][:] = fcst2_ar
      fid.close()
   else:
      """ Create calibration curve """
      I = np.where((np.isnan(obs) == 0) & (np.isnan(fcst) == 0))[0]
      if args.y is not None:
         x, y, = method.get_single_curve(obs[I], fcst[I], args.y)
         mpl.plot(x, y, 'k-o')
         #q = [np.min(x), np.max(x)]
         #mpl.plot(q, q, '-', color="gray", lw=2)
         #mpl.gca().set_aspect(1)
         mpl.grid()
         mpl.show()
      else:
          import time as timing
          s = timing.time()
          x, y, = method.get_curve(obs[I], fcst[I], np.min(fcst[I]), np.max(fcst[I]))
          print timing.time() - s
          if args.curve_file is not None:
             file = open(args.curve_file, 'w')
             file.write("unixtime obs fcst\n")
             for i in range(len(x)):
                file.write("%d %f %f\n" % (1e9 + i,  x[i], y[i]))
             file.close()
          else:
             # x, y, lower, upper = method.get_curve(obs[I], fcst[I], np.min(fcst[I]), np.max(fcst[I]))
             #mpl.plot(fcst[I], obs[I], 'r.', alpha=0.3)
             mpl.plot(x, y, 'k-o')
             # mpl.plot(lower, y, 'k--o')
             # mpl.plot(upper, y, 'k--o')
             # mpl.plot((lower+upper)/2, y, 'k--o')
             mpl.plot([-15, 15], [-15, 15], '-', lw=2, color="gray")
             mpl.grid()
             mpl.gca().set_aspect(1)
             mpl.xlim([-15, 15])
             mpl.ylim([-15, 15])
             mpl.show()


if __name__ == '__main__':
   main()
