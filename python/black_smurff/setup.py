from setuptools import setup

setup(name='black_smurff',
        version='0.4',
        description='Wrapper around macau to act like smurff',
        url='http://github.com/ExaScience/smurff',
        author='Tom Vander Aa',
        author_email='tom.vanderaa@imec.be',
        license='MIT',
        packages=[],
        install_requires=[ 'macau', 'scipy' ],
        scripts=['bin/smurff'],
        zip_safe=False)
