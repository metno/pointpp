import sys
import numpy as np
import pointpp.driver
import pointpp.gen
import pointpp.radpro


def main():
    pointpp.driver.run(sys.argv)

def pointgen():
    pointpp.gen.run(sys.argv)

def pointradpro():
    pointpp.radpro.run()

if __name__ == '__main__':
    main()
