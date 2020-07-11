import sys

DEBUG = False

def progress_bar(fraction, width, text=""):
   num_x = int(fraction * (width-len(text) - 2))
   num_space = width - num_x - len(text) - 2
   sys.stdout.write("\r" + text + "[" + "X" *  num_x + " " * num_space + "]")
   sys.stdout.flush()


def error(message):
   """ Write error message to console and abort """
   print("\033[1;31mError: " + message + "\033[0m")
   sys.exit(1)


def debug(message):
   if DEBUG:
      print(message)


def warning(message):
   """ Write a warning message to console """
   print("\033[1;33mWarning: " + message + "\033[0m")
