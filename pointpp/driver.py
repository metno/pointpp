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
   parser.add_argument('-b', type=int, default=100, metavar="NUM", help="Number of bins", dest="num_bins")
   parser.add_argument('-o', metavar="FILE", help="Output filename", dest="ofile")
   parser.add_argument('-m', metavar="METHOD", help="Optimization method", required=True, dest="method")
   parser.add_argument('-loc', help="Post-process each station independently?", dest="location_dependent", action="store_true")
   parser.add_argument('-lt', help="Post-process each leadtime independently?", dest="leadtime_dependent", action="store_true")
   parser.add_argument('-tt', type=int, help="Training time", dest="ttime")

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
      method = pointpp.method.MyMethod(metric, nbins=args.num_bins, monotonic=True)

   D = obs_ar.shape[0]
   LT = obs_ar.shape[1]
   LOC= obs_ar.shape[2]

   if args.ofile is not None:
      shutil.copyfile(args.file, args.ofile)
      if args.method == "pers" or args.method == "fpers":
         """ Persistence methods should always be location and date independent """
         fcst2_ar = np.nan * np.zeros(obs_ar.shape)
         for i in range(D):
            for j in range(LOC):
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
            for i in range(LT):
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
      x, y = method.get_curve(obs, fcst, np.nanmin(fcst), np.nanmax(fcst))
      mpl.plot(fcst, obs, 'r.', alpha=0.3)
      mpl.plot(x, y, 'k-o')
      #mpl.plot([-10, 10], [-10, 10], '-', lw=2, color="gray")
      mpl.gca().set_aspect(1)
      #mpl.xlim([-10, 10])
      #mpl.ylim([-10, 10])
      mpl.show()


if __name__ == '__main__':
   main()
