Software for post-processing weather forecasts at points
========================================================

Pointpp is a command-line tool to post-process weather forecasts for point locations. It uses input files prepared for Verif (https://github.com/WFRT/verif).

For more information about how to use Pointpp, checkout the wiki page: https://github.com/metno/pointpp/wiki

Features
--------

Pointpp supports several post-processing methods such as:

* Quantile-mapping
* Linear regression
* Conditional mean
* Climatology
* Persistence
* Score-optimizer (such as ETS-optimizer)

Installing on Ubuntu
--------------------

Clone this repository and then run the following command inside the extracted folder:

.. code-block:: bash

  sudo pip install -e .

This will create the executable ``/usr/local/bin/pointpp``. Add ``/usr/local/bin`` to your PATH environment
variable if necessary. If you do not have sudo privileges do:

.. code-block:: bash

  pip install -e . --user

This will create the executable ``~/.local/bin/pointpp``. Add ``~/.local/bin`` to your PATH environment
variable.

Copyright and license
---------------------

Copyright © 2017 Thomas Nipen. Pointpp is licensed under the 3-clause BSD license. See LICENSE file.
