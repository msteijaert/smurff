# Use an official Python runtime as a base image
FROM jupyter/scipy-notebook

MAINTAINER Tom Vander Aa <tom.vanderaa@imec.be>

USER root

RUN apt-get update && \
    apt-get -yq install libopenblas-dev autoconf gfortran \
                libboost-system-dev libboost-program-options-dev  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install cmake
RUN mkdir $HOME/cmake && \
    cd $HOME/cmake && \
    wget https://cmake.org/files/v3.10/cmake-3.10.2-Linux-x86_64.sh && \
    bash cmake-3.10.2-Linux-x86_64.sh --skip-license --prefix=/usr/local && \
    cd $HOME && rm -rf cmake


# Install Eigen3.3.4
RUN mkdir $HOME/eigen3 && \
    cd $HOME/eigen3 && \
    wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz && \
    tar xzf 3.3.4.tar.gz && \
    cd eigen-eigen-* && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j && \
    make install && \
    cd $HOME && rm -rf eigen3

# Install smurff w/o python
RUN mkdir $HOME/git && \
    cd $HOME/git && \
    git clone https://github.com/ExaScience/smurff.git && \
    cd $HOME/git/smurff && \
    git checkout v0.10.2 && \
    mkdir build && \
    cd build && \
    cmake ../lib/smurff-cpp/cmake -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j && \
    make install

# Install python wrapper for smurff
RUN cd $HOME/git/smurff/python/smurff && \
    sed -i -e "s/+ lapack_opt_info\['libraries'\]/ + ['openblas']/" setup.py && \
    sed -i -e "s/+ lapack_opt_info\['include_dirs'\]//" setup.py && \
    python3 setup.py install --with-openmp --with-smurff-cpp=/usr/local && \
    cd $HOME && rm -rf git


ADD smurff.ipynb $HOME/work/

