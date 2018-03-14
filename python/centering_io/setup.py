from setuptools import setup

setup(name='centering_io',
        version='0.1',
        description='Centering sparse and dense matrices in binary format',
        url='http://github.com/ExaScience/smurff',
        author='Tom Vander Aa',
        author_email='tom.vanderaa@imec.be',
        license='MIT',
        packages=['centering_io'],
        install_requires=[ 'scipy', 'numpy' ],
        zip_safe=False)
