#!/usr/bin/env python3

import os
import glob
import pandas as pd
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser(description='Compute AUC from CSV file')
parser.add_argument('csv_file', metavar='FILE', nargs=1)
parser.add_argument('--env', metavar='NAME')
parser.add_argument('--expected', metavar='FILE', default='expected-pass.csv')
parser.add_argument('--fail', metavar='FILE', default='expected-fail.csv')
parser.add_argument('--approx', metavar='NUM', default=0.05)
parser.add_argument('--verbose', action="store_true")
args = parser.parse_args()

expected_pass = pd.read_csv(args.expected).set_index("name").to_dict(orient = "index")
expected_fail = pd.read_csv(args.fail).set_index( ["env", "name" ]).to_dict(orient = "index")

def compare(actual, expected, verbose = False):
    max_deviation = actual * args.approx
    lower_bound = expected - max_deviation
    upper_bound = expected + max_deviation
    passed = (actual > lower_bound and actual < upper_bound)
    if verbose:
        print("    comparison %.2f < %.2f < %.2f %s" % ( lower_bound, actual, upper_bound, "passed" if passed else "failed") )


    return passed


num_tests = 0
num_skipped = 0
num_total = 0
num_expected_pass = 0
num_expected_fail = 0
num_unexpected_pass = 0
num_unexpected_fail = 0

for f in args.csv_file:
    entries = pd.read_csv(f).to_dict(orient = "index")
    for t in entries.values():
        num_total += 1
        state = None
        # skip if not in envs
        if args.env and t["env"] not in args.env:
            num_skipped += 1
            if verbose:
                print(t["name"] + " not in env " + args.envs)
            continue

        # warn if not in reference
        if t["name"] not in expected_pass:
            num_skipped += 1
            if verbose:
                print(t["name"] + " not found in " + args.reference)
            continue

        if args.verbose:
            print("%s / %s :" % (t["env"], t["name"]))

        does_pass = compare(t["rmse"], expected_pass[t["name"]]["rmse"], args.verbose)
        does_pass = compare(t["auc"], expected_pass[t["name"]]["auc"], args.verbose) and does_pass
        in_fail = (t["env"], t["name"]) in expected_fail

        if (not does_pass and not in_fail):
            state = "UNEXPECTED FAIL"
            num_unexpected_fail += 1
        elif (not does_pass and in_fail):
            state = "EXPECTED FAIL"
            num_expected_fail +=1 
        elif (does_pass and not in_fail):
            num_expected_pass += 1
            state = "EXPECTED PASS"
        else:
            num_unexpected_pass += 1
            state = "UNEXPECTED PASS"

        num_tests += 1

        if args.verbose:
            print("    %s" % (state,))

assert num_unexpected_pass + num_expected_pass + num_unexpected_fail + num_expected_fail == num_tests
assert num_tests + num_skipped == num_total 


if (num_unexpected_pass == 0 and num_unexpected_fail == 0):
    print("All tests passed (%d tests, %d expected pass, %d expected fails)." % (num_tests, num_expected_pass, num_expected_fail))
else:
    if (num_unexpected_fail != 0):
        print("%d tests out of %d tests unexpectely FAILED." % ( num_unexpected_fail, num_tests ))
    if (num_unexpected_pass != 0):
        print("%d tests out of %d tests unexpectely PASSED." % ( num_unexpected_pass, num_tests ))
    print("(Rerun with --verbose to see which tests failed)")

if (num_skipped > 0):
    print("  %d test skipped" % num_skipped)
