# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
.PHONY: help clean html 

all: html

images: notebooks/tensor-model.png

notebooks/%.png: ../python/notebooks/%.png
	cp $? notebooks/

notebooks/%.rst: ../python/notebooks/%.ipynb
	jupyter nbconvert --to rst --output-dir=notebooks $?
	
notebooks: notebooks/a_first_example.rst notebooks/different_methods.rst

html: images notebooks
	sphinx-build -b html .  _build/html

clean:
	rm -r _build notebooks/* 
