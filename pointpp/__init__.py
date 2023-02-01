import sys
import numpy as np
import pointpp.gen


__version__ = "0.2.0"

def main():
    import pointpp.__main__

    pointpp.__main__.main()

def pointgen():
    pointpp.gen.run(sys.argv)

if __name__ == "__main__":
    main()
