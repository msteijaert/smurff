#!/bin/bash

set -e

VERSION=0.4.3
DEBUG=0
DIR=

while [[ $# -gt 1 ]]
do
    key="$1"

    case $key in
        --version)
            VERSION="$2"
            shift # past argument
            ;;
        --dir)
            DIR="$2"
            shift # past argument
            ;;
        --debug)
            DEBUG=1
            shift # past argument
            ;;
         *)
            # unknown option
            ;;
    esac
    shift # past argument or value
done


test -z "$DIR" && DIR=$(mktemp -d -t macau-v${VERSION}-)
mkdir -p $DIR

echo "Installing macau $VERSION in $DIR"

cd $DIR

# virtualenv first
virtualenv penv

. penv/bin/activate

pip install numpy scipy pandas Cython
pip install jupyter 

rm -rf macau
git clone https://github.com/jaak-s/macau
cd macau

if [ "$DEBUG" -eq 1 ]; then
    set -i -e 's/-O3/-O0/g' setup.py
fi

git checkout v$VERSION

python setup.py install


