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
   parser.add_argument('file', help="Input file", nargs="?")
   parser.add_argument('-o', metavar="FILE", help="Output filename", dest="ofile")
   parser.add_argument('-m', metavar="METHOD", help="Optimization method", required=True, dest="method")
   parser.add_argument('-loc', help="Post-process each station independently?", dest="location_independent", action="store_true")
   parser.add_argument('-lt', help="Post-process each leadtime independently?", dest="leadtime_independent", action="store_true")
   parser.add_argument('-tt', type=int, help="Training time", dest="ttime")

   args = parser.parse_args()

   if args.file is not None:
       input = verif.input.get_input(args.file)
       obs = input.obs.flatten()
       fcst = input.fcst.flatten()
       obs_ar = input.obs
       fcst_ar = input.fcst
   else:
      N = 100000
      sigma = 2
      control = np.linspace(-15,15, N)
      obs = control + 0*np.random.normal(0, sigma, N)
      fcst = control + np.random.normal(0, sigma, N)

   method = pointpp.method.get(args.method)
   if method is None:
       metric = verif.metric.get(args.method)
       if metric is None:
           verif.util.error("Could not understand '%s'" % args.method)
       method = pointpp.method.MyMethod(metric, nbins=100, monotonic=True)

   if args.file is not None and args.ofile is not None:
      shutil.copyfile(args.file, args.ofile)
      if args.method == "persistence" or args.method == "forecastpersistence":
          fcst2_ar = np.nan * np.zeros(obs_ar.shape)
          # if len(obs_ar.shape) == 1:
          #     np.expand_dims(obs_ar, 1)
          # if len(obs_ar.shape) == 2:
          #     np.expand_dims(obs_ar, 2)
          for i in range(obs_ar.shape[0]):
              for j in range(obs_ar.shape[2]):
                  fcst2_ar [i, :, j] = method.calibrate(obs_ar[i, :, j], fcst_ar[i, :, j], fcst_ar[i, :, j])
      else:

          if args.location_independent and args.leadtime_independent:
              fcst2_ar = np.nan * np.zeros(obs_ar.shape)
              for i in range(obs_ar.shape[1]):
                  for j in range(obs_ar.shape[2]):
                      fcst2_ar[:, i, j] = method.calibrate(obs_ar[:, i, j], fcst_ar[:, i, j], fcst_ar[:, i, j])
          else:
              fcst2 = method.calibrate(obs, fcst, fcst)
              fcst2_ar = np.reshape(fcst2, fcst_ar.shape)
      fid = netCDF4.Dataset(args.ofile, 'a')
      print "Writing"
      fid.variables["fcst"][:] = fcst2_ar
      fid.close()
   else:
      x, y = method.get_curve(obs, fcst, -30, 30)
      mpl.plot(fcst, obs, 'r.', alpha=0.3)
      mpl.plot(x, y, 'k-o')
      mpl.plot([-10, 10], [-10, 10], '-', lw=2, color="gray")
      mpl.gca().set_aspect(1)
      mpl.xlim([-10, 10])
      mpl.ylim([-10, 10])
      mpl.show()


if __name__ == '__main__':
   main()
