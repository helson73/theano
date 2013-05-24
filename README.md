theano
======


Test system:
Ubuntu 12.04LTS 64bit

Prerequisite:
$ sudo apt-get install git python-dev gfortran build-essential python-pip

[Prerequisite for GPU computing]
$ sudo apt-add-repository ppa:ubuntu-x-swat/x-updates
$ sudo apt-get update
$ sudo apt-get install nvidia-current
$ wget http://developer.download.nvidia.com/compute/cuda/5_0/rel-update-1/installers/cuda_5.0.35_linux_64_ubuntu11.10-1.run
$ sudo chmod +x cuda_5.0.35_linux_64_ubuntu11.10-1.run
$ sudo ./cuda_5.0.35_linux_64_ubuntu11.10-1.run

Install OpenBlas:
$ git clone git://github.com/xianyi/OpenBLAS
$ make FC=gfortran
$ sudo make PREFIX=/usr/local/ install

Relocate symbolic link:
$ cd /usr/local/lib
$ ln -s libopenblas.so /usr/lib/libblas.so
$ ln -s libopenblas.so.0 /usr/lib/libblas.so.3gf
$ cd /usr/lib/lapack
$ ln -s liblapack.so.3gf /usr/lib/liblapack.so.3gf

Install Numpy:
$ git clone https://github.com/numpy/numpy
$ python setup.py build
$ sudo python setup.py install

Install Scipy:
$ git clone https://github.com/scipy/scipy
$ python setup.py build
$ sudo python setup.py install

Install Theano:
$ sudo pip install Theano
$ cd /usr/local/lib/python2.7/dist-packages/theano/
$ vi blas.py
  - then add 'import numpy.distutils.__config__' just below the code 'import numpy.distutils', line 134

Set environment value:
$ cd
$ vi .bashrc
  - then add 'OPENBLAS_NUM_THREADS=n' to the last line. 
  - Where n is the maximum number of threads that your CPU supports. 
  - e.g. for Core i5 2500K, OPENBLAS_NUM_THREADS=4, for Xeon E5-2667, OPENBLAS_NUM_THREADS=24
$ vi .theanorc 
  * just for GPU computing, if you just use CPU, this step could be passed *
  - then add following lines:  
    [cuda]
    root = /usr/local/cuda-5.0/bin
    
    [global]
    device = gpu
    floatX = float32
    allow_gc = False 

    [blas]
    ldflags = -lopenblas

Test:
$ python
>>> import numpy
>>> import scipy
>>> import theano
* after typing three codes above, if any error occurred, 
* you should reconsider once again whether previous steps 
* were excuted correctly.

Test_Code:
-----------------------------------------------
'''
Author: Jianri Li
This code will test matrix dot operation of two 10000x10000 matrices three times.
'''
import numpy as np
import theano
import theano.tensor as T
import time
rng = np.random

N = 10000
k = T.iscalar("k")
A = T.matrix("A")
D = rng.randn(N, N).astype(theano.config.floatX)
result, updates = theano.scan(fn=lambda prior, A: T.dot(prior, A), non_sequences=A, outputs_info=T.identity_like(A), n_steps=k)

final_result = result[-1]

power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

t0 = time.time()
a = power(D,3)
t1 = time.time()
print a
print 'Cost time: ', t1 - t0
-----------------------------------------------
create test.py and add codes above, then excute:
for CPU test:
$ THEANO_FLAGS=mode=FAST_RUN,device=cpu python test.py 
for GPU test:
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu python test.py

Rough results for common devices:
Geforce GTX 680 4GB @ 1.17Ghz: Cost time: 4.5s
Core i5 2500K @ 3.8Ghz with 4 threads: Cost time: 39s
Xeon E5-2667 @ 2.93Ghz with 12 threads: Cost time: 33s
