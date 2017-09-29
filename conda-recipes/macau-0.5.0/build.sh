sed -i -e 's/library_dirs = ldirs/library_dirs = []/g' setup.py
$PYTHON setup.py install     # Python command to install the script.
