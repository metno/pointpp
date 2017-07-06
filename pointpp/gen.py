import argparse
import sys
import numpy as np
import shutil
import netCDF4
import verif.util
import pointpp.version


def run(argv):
   parser = argparse.ArgumentParser(prog="pointgen", description="Creates random observations and forecasts")
   parser.add_argument('--version', action="version", version=pointpp.version.__version__)
   parser.add_argument('file', help="Output file", nargs=1)
   parser.add_argument('-n', metavar="NUM", type=int, help="Number of points", required=True, dest="num")
   parser.add_argument('-s', metavar="SIGMA", type=float, help="Standard deviation", dest="sigma")
   parser.add_argument('-so', metavar="SIGMA", type=float, help="Observation standard deviation", dest="sigma_obs")
   parser.add_argument('-sf', metavar="SIGMA", type=float, help="Forecast standard deviation", dest="sigma_fcst")
   parser.add_argument('-r', metavar="LOWER,UPPER", type=verif.util.parse_numbers, help="Range", required=True, dest="range")
   parser.add_argument('-seed', metavar="NUM", type=int, help="Random seed", dest="seed")

   if len(sys.argv) == 1:
      parser.print_help()
      sys.exit(1)
   args = parser.parse_args()
   if args.seed is not None:
      np.random.seed(args.seed)

   N = args.num
   if args.sigma is None:
      sigma_obs = args.sigma_obs
      sigma_fcst = args.sigma_fcst
   else:
      sigma_obs = args.sigma
      sigma_fcst = args.sigma
      
   control = np.linspace(args.range[0], args.range[1], N)
   if sigma_obs > 0:
      obs = control + np.random.normal(0, sigma_obs, N)
   else:
      obs = control
   if sigma_fcst > 0:
      fcst = control + np.random.normal(0, sigma_fcst, N)
   else:
      fcst = control

   fid = netCDF4.Dataset(args.file[0], 'w')
   fid.createDimension("time")
   fid.createDimension("location", 1)
   fid.createDimension("leadtime", 1)
   var_time = fid.createVariable("time", "i4", ["time"])
   var_leadtime = fid.createVariable("leadtime", "i4", ["leadtime"])
   var_location = fid.createVariable("location", "i4", ["location"])
   var_lat = fid.createVariable("lat", "f4", ["location"])
   var_lon = fid.createVariable("lon", "f4", ["location"])
   var_alt = fid.createVariable("altitude", "f4", ["location"])
   var_obs = fid.createVariable("obs", "f4", ["time", "leadtime", "location"])
   var_fcst = fid.createVariable("fcst", "f4", ["time", "leadtime", "location"])
   var_time[:] = np.arange(0, N)
   var_obs[:] = obs
   var_fcst[:] = fcst
   var_leadtime[:] = [0]
   var_lat[:] = [0]
   var_lon[:] = [0]
   var_alt[:] = [0]
   var_location[:] = [0]
   fid.close()


if __name__ == '__main__':
   main()
