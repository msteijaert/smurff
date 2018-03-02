from setuptools import setup

setup(name='matrix_io',
        version='0.5',
        description='Reading and writing sparse and dense matrices in binary format',
        url='http://github.com/ExaScience/smurff',
        author='Tom Vander Aa',
        author_email='tom.vanderaa@imec.be',
        license='MIT',
        packages=['matrix_io'],
        install_requires=[ 'scipy' ],
        scripts=['bin/csv2ddm', 'bin/mtx2ddm', 'bin/mtx2sbm', 'bin/mtx2sdm' ],
        zip_safe=False)
