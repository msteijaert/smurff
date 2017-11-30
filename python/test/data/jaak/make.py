#!/usr/bin/env python

import urllib.request
import scipy.io as sio
import os

urls = [
        "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm",
        "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm",
        "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compounds.csv",
        "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-proteins-uniprot.csv",
]


for url in urls:
    output = os.path.basename(url)
    urllib.request.urlretrieve(url, output)

# binary for probit
ic50 = sio.mmread("chembl-IC50-346targets.mm")
ic50.data = (ic50.data >= 6) * 1.
with open("chembl-IC50-346targets-binary.mm", "wb") as f:
    sio.mmwrite(f, ic50)
