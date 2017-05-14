import sys
import numpy as np
import pointpp.driver
import pointpp.gen


def main():
   pointpp.driver.run(sys.argv)

def pointgen():
   pointpp.gen.run(sys.argv)

if __name__ == '__main__':
   main()
