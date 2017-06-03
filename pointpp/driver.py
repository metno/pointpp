import argparse
import numpy as np
import shutil
import verif.input
import verif.metric
import pointpp.version
import pointpp.method
import matplotlib.pyplot as mpl
import netCDF4


def run(argv):
   parser = argparse.ArgumentParser(prog="ppverif", description="Hybrid weather generator, combining stochastic and physical modelling")
   parser.add_argument('--version', action="version", version=pointpp.version.__version__)
   parser.add_argument('file', help="Input file")
   parser.add_argument('-t', help="Training file", dest="file_training")
   parser.add_argument('-b', type=int, default=100, metavar="NUM", help="Number of bins", dest="num_bins")
   parser.add_argument('-o', metavar="FILE", help="Output filename", dest="ofile")
   parser.add_argument('-m', metavar="METHOD", help="Optimization method", required=True, dest="method")
   parser.add_argument('-loc', help="Post-process each station independently?", dest="location_dependent", action="store_true")
   parser.add_argument('-lt', help="Post-process each leadtime independently?", dest="leadtime_dependent", action="store_true")
   parser.add_argument('-tt', type=int, help="Training time", dest="ttime")

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
      if metric is None:
         verif.util.error("Could not understand '%s'" % args.method)
      method = pointpp.method.MyMethod(metric, nbins=args.num_bins, monotonic=True)

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
         e2t_loc[Ie[0]] = It[0]
   D = eobs.shape[0]
   LT = eobs.shape[1]
   LOC = eobs.shape[2]

   if args.ofile is not None:
      shutil.copyfile(args.file, args.ofile)
      if args.method == "pers" or args.method == "fpers":
         """ Persistence methods should always be location and date dependent """
         efcst2 = np.nan * np.zeros(eobs.shape)
         for i in range(D):
            for j in e2t_loc:
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
      fid.close()
   else:
      I = np.where((np.isnan(obs) == 0) & (np.isnan(fcst) == 0))[0]
      x, y = method.get_curve(obs[I], fcst[I], np.min(fcst[I]), np.max(fcst[I]))
      mpl.plot(fcst[I], obs[I], 'r.', alpha=0.3)
      mpl.plot(x, y, 'k-o')
      #mpl.plot([-10, 10], [-10, 10], '-', lw=2, color="gray")
      mpl.gca().set_aspect(1)
      #mpl.xlim([-10, 10])
      #mpl.ylim([-10, 10])
      mpl.show()


if __name__ == '__main__':
   main()
