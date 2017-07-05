import argparse
import numpy as np
import shutil
import sys
import matplotlib.pyplot as mpl
import netCDF4
import verif.input
import verif.metric
import pointpp.util
import pointpp.version
import pointpp.method
import pointpp.metric


def run():
   methods = [x[0].lower() for x in pointpp.method.get_all()]
   parser = argparse.ArgumentParser(prog="radpro_pointpp", description="Post-processing using radpro algorithm")
   parser.add_argument('--version', action="version", version=pointpp.version.__version__)
   parser.add_argument('--debug', help="Show debug information", action="store_true")
   parser.add_argument('file', help="Verif NetCDF input file")
   parser.add_argument('-t', metavar="FILE", help="Verif NetCDF file to use for Training ", dest="file_training")
   parser.add_argument('-o', metavar="FILE", help="Output filename", dest="ofile")
   parser.add_argument('-f', metavar="FILE", help="Save image to this filename", dest="imagefile")
   # parser.add_argument('-delay', metavar="FILE", help="Verif NetCDF file to use for Training ", dest="file_training")

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

   method = pointpp.method.Radpro()
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
   method = pointpp.radpro.Radpro()

   """ Train the method """
   param1, param2 = method.train(tobs, tfcst)

   if args.ofile is not None:
      """ Create output """
      shutil.copyfile(args.file, args.ofile)
      efcst2 = np.nan * np.zeros(eobs.shape)

      """ Apply corrections """
      efcst2 = method.apply(eobs, efcst)

      fid = netCDF4.Dataset(args.ofile, 'a')
      print "Writing"
      fid.variables["fcst"][:] = efcst2
      if eobs2 is not None:
         fid.variables["obs"][:] = eobs2
      fid.close()
   else:
      import matplotlib.pylab as mpl
      import matplotlib.ticker
      mpl.plot(range(tobs.shape[1]), param1, 'o-', color="b", lw=2, label="Weight recent bias")
      mpl.plot(range(tobs.shape[1]), param2, 'o-', color="r", lw=3, label="Weight yesterday's bias")
      locator = matplotlib.ticker.MultipleLocator(24)
      mpl.gca().xaxis.set_major_locator(locator)
      mpl.xlabel("Lead time")
      mpl.xlabel("Weight")
      mpl.ylim([0, 1])
      mpl.legend()
      if args.imagefile is not None:
         mpl.savefig(args.imagefile)
      else:
         mpl.show()


if __name__ == '__main__':
   run()
