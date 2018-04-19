from .prepare import make_train_test

import tempfile
import urllib.request
import scipy.io as sio
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
        ]
def download_chembl():
    with tempfile.TemporaryDirectory() as tmpdirname:
        for url, expected_sha, output in urls:
            urllib.request.urlretrieve(url, output)
            actual_sha = sha256(open(output, "rb").read()).hexdigest()
            assert actual_sha == expected_sha

        ic50 = sio.mmread("chembl-IC50-346targets.mm")
        feat = sio.mmread("chembl-IC50-compound-feat.mm")

        ## creating train and test sets
        ic50_train, ic50_test = make_train_test(ic50, 0.2)

        return (ic50_train, ic50_test, feat)