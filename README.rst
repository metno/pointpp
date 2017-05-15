Point post-processing software
==============================

Pointpp is a command-line tool to post-process weather forecasts for point locations. It uses input
files prepared for Verif (https://github.com/WFRT/verif).

For more information, checkout the wiki page: https://github.com/WFRT/verif/wiki

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

**Prerequisites**

Pointpp requires Verif to be installed.

**Installing from source**

Download the source code of the latest version: https://github.com/tnipen/pointpp/releases/. Unzip the
file and navigate into the extracted folder.

Then install Pointpp by executing the following inside the extracted folder:

.. code-block:: bash

  sudo pip install -e .

This will create the executable ``/usr/local/bin/verif``. Add ``/usr/local/bin`` to your PATH environment
variable if necessary. If you do not have sudo privileges do:

.. code-block:: bash

  pip install -e . --user

This will create the executable ``~/.local/bin/verif``. Add ``~/.local/bin`` to your PATH environment
variable.

Copyright and license
---------------------

Copyright © 2017 Thomas Nipen. Pointpp is licensed under the 3-clause BSD license. See LICENSE file.
