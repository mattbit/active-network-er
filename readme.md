# ER active flow network simulations

[![Build Status](https://travis-ci.com/mattbit/active-network-er.svg?token=zKpBnjBx4d1NEMb7zFbd&branch=master)](https://travis-ci.com/mattbit/active-network-er)

The code provides a framework to implement simulations of particle motion on networks. Different models can be used for the network and the particle behaviour. In particular, the code provides an implementation of the active-flow network model to describe motion on the endoplasmic reticulum.

Some examples of network structures are provided in graphml format in `resources/graphs`.

The code has been tested on Python 3.5, 3.6, 3.7.


#### Active-flow network generates molecular transport by packets

The code to reproduce the figures of “Active flow network generates molecular transport by packets” (Proceedings of the Royal Society B, 2020) by Matteo Dora & David Holcman is provided in the `scripts_prsb` subfolder.
Each script will run the required simulations and produce the corresponding figures (it can take some time!).

### Documentation

Documentation can be generated with sphinx using the scripts provided in the `docs` folder.

Don't hesitate to reach out matteo.dora@ens.psl.eu if you have troubles running the code, if you have doubts or need any help.

### Development

Development is done using `pipenv` to manage dependencies. Test can be run with python `unittest` (`$ python -m unittest discover test`) or `pytest`.

### Authors

The code was written by Matteo Dora (matteo.dora@ens.psl.eu) at the Biology Department of the École Normale Supérieure in Paris.

### License

The code is released with GPLv3 license.
