#!/usr/bin/env python

import urllib.request
import scipy.io as sio
import os
from hashlib import sha256

urls = [
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm",
            "10c3e1f989a7a415a585a175ed59eeaa33eff66272d47580374f26342cddaa88",
            "chembl-IC50-346targets.mm",
            ),
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm",
            "f9fe0d296272ef26872409be6991200dbf4884b0cf6c96af8892abfd2b55e3bc",
            "chembl-IC50-compound-feat.mm", 
            ),
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compounds.csv",
            "e8f045a67ee149c6100684e07920036de72583366596eb5748a79be6e3b96f7c",
            "chembl-IC50-compounds.csv",
            ),
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-proteins-uniprot.csv",
            "224b1b44abcab8448b023874f4676af30d64fe651754144f9cbdc67853b76ea8",
            "chembl-IC50-proteins-uniprot.csv",
            ),
        ]


for url, expected_sha, output in urls:
    if os.path.isfile(output):
        actual_sha = sha256(open(output, "rb").read()).hexdigest()
        if (expected_sha == actual_sha):
            continue

    print("download %s" % output)
    urllib.request.urlretrieve(url, output)

ic50 = sio.mmread("chembl-IC50-346targets.mm")

# 0,1 binary for probit
ic50_01 = ic50.copy()
ic50_01.data = (ic50_01.data >= 6) * 1.
sio.mmwrite(open("chembl-IC50-346targets-01.mm", "wb"), ic50_01)

# -1,+1
ic50_11 = ic50.copy()
ic50_11.data = ((ic50.data >= 6) * 2.) - 1.
sio.mmwrite(open("chembl-IC50-346targets-11.mm", "wb"), ic50_11)
