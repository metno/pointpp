import sys


def progress_bar(fraction, width, text=""):
   num_x = int(fraction * (width-len(text) - 2))
   num_space = width - num_x - len(text) - 2
   sys.stdout.write("\r" + text + "[" + "X" *  num_x + " " * num_space + "]")
   sys.stdout.flush()
