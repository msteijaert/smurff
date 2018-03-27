# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
.PHONY: help clean html 

html: 
	jupyter nbconvert --to rst --output-dir=source ../python/jupyter-notebook/smurff.ipynb
	sphinx-build -b html .  _build/html

clean:
	rm -rf $(BUILDDIR)/*
