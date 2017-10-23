#!/usr/bin/env python

import loom.client as lc
import argparse
import glob

parser = argparse.ArgumentParser(description='Loom client for running SMURFF tests')

parser.add_argument('--testdir',  metavar='DIR', dest='dir', nargs=1, help='Output dir', default = 'work/latest')
parser.add_argument('--host',  metavar='HOSTNAME', dest='loom_server', nargs=1, help='Loom server HOST', default = 'localhost')
parser.add_argument('--port',  metavar='PORTNUMBER', dest='loom_port', nargs=1, help='Loom server PORT', default = 9010)
parser.add_argument('--cmd',  metavar='FILE', dest='cmd', nargs=1, help='Scripts to look for', default = 'cmd')

args = parser.parse_args()


def main():
    tasks = []
    for f in glob.iglob('%s/**/%s' % (args.dir, args.cmd), recursive=True):
        task = lc.tasks.run("bash -e %s" % f)
        tasks.append(task)
    c = lc.Client(args.loom_server, args.loom_port)
    futures = c.submit(tasks)
    results = c.gather(futures)
    print(results)

if __name__ == "__main__":
    main()
