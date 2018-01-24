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
feat = sio.mmread("chembl-IC50-compound-feat.mm")

def make_01():
    # 0,1 binary for probit
    ic50_01 = ic50.copy()
    ic50_01.data = (ic50_01.data >= 6) * 1.
    sio.mmwrite(open("chembl-IC50-346targets-01.mm", "wb"), ic50_01)

def make_11():
    # -1,+1
    ic50_11 = ic50.copy()
    ic50_11.data = ((ic50.data >= 6) * 2.) - 1.
    sio.mmwrite(open("chembl-IC50-346targets-11.mm", "wb"), ic50_11)

def make_100compounds():
    ic50_100c = ic50.tocsr()[0:100,:]
    sio.mmwrite(open("chembl-IC50-346targets-100compounds.mm", "wb"), ic50_100c)

def feat_100():
    feat_100 = feat.tocsr()[0:100,:]
    feat_100 = feat_100[:,feat_100.getnnz(0)>0]
    return feat_100

def make_feat():
    sio.mmwrite(open("chembl-IC50-100compounds-feat.mm", "wb"), feat_100())

def make_feat_dense():
    sio.mmwrite(open("chembl-IC50-100compounds-feat-dense.mm", "wb"), feat_100().todense())

generated_files = [
        ( "7c3a1a381a2017a463f201100ee5b257b7ee01819f41f5737f4a817411beaef7",
            "chembl-IC50-100compounds-feat-dense.mm",
            make_feat_dense,
            ),
        ( "30253d120e06f0ab9e766185ca2991fff3a500c6c51064ed94034c057d676842",
            "chembl-IC50-100compounds-feat.mm",
            make_feat,
            ),
        ( "3114693bab82bdf7ac4b58fde0007bc2291f2c52e0de0c0a7f0e1a332010bdfe",
            "chembl-IC50-346targets-01.mm",
            make_01,
            ),
        ( "66cc294bdb13f8d6900cfd467b75f15c09e4808c93f2f8e7aeb55e37956c4df6",
            "chembl-IC50-346targets-100compounds.mm",
            make_100compounds,
            ),
        ( "2aa07a0e0aed3a1cb0bf17fed9aabcdefbd57ed0897908e9be50aa58fc082515",
            "chembl-IC50-346targets-11.mm",
            make_11,
            ),
        ]

for expected_sha, output, func in generated_files:
    if os.path.isfile(output):
        actual_sha = sha256(open(output, "rb").read()).hexdigest()
        if (expected_sha == actual_sha):
            continue

    print("make %s" % output)
    func()

    actual_sha = sha256(open(output, "rb").read()).hexdigest()
    assert (expected_sha == actual_sha)


