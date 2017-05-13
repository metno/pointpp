import argparse
import numpy as np
import shutil
import verif.input
import verif.metric
import pointpp.version
import pointpp.method
import matplotlib.pyplot as mpl


def run(argv):
   parser = argparse.ArgumentParser(prog="ppverif", description="Hybrid weather generator, combining stochastic and physical modelling")
   parser.add_argument('--version', action="version", version=pointpp.version.__version__)
   parser.add_argument('file', help="Input file", nargs="?")
   parser.add_argument('-o', metavar="FILE", help="Output filename", required=True, dest="ofile")
   parser.add_argument('-c', metavar="METHOD", help="Method", required=True, dest="method")
   parser.add_argument('-m', metavar="METRIC", help="Optimization metric", dest="metric")

   args = parser.parse_args()

   if args.file is not None:
       input = verif.input.get_input(args.file[0])
       obs = input.obs.flatten()
       fcst = input.fcst.flatten()
   else:
      N = 1000
      sigma = 2
      control = np.linspace(-5,5, N)
      obs = control + np.random.normal(0, sigma, N)
      fcst = control + np.random.normal(0, sigma, N)

   if args.method == "mymethod":
       metric = verif.metric.get(args.metric)
       method = pointpp.method.MyMethod(metric, nbins=100, monotonic=False)
   else:
       method = pointpp.method.get(args.method)

   x, y = method.get_curve(obs, fcst, -30, 30)
   mpl.plot(x, y, 'k-o')
   mpl.show()
   if args.file is not None and args.ofile is not None:
      shutil.copyfile(args.file, args.ofile)


if __name__ == '__main__':
   main()
