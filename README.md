# Advanced computer graphics(NRG) Seminar for FRI(https://fri.uni-lj.si) 2018/2019

##### Requirements:
- ability to compile cython(https://cython.org/)
-- visual studio tools for python development, C compiler etc..
- ability to run python

##### Usage:
- Run 'main.py'. If you wish to input your own image or change the params, edit the code as you wish. Cython code is regenerated on every run if it has changed (by pyximport.
- To compile just the cython code run 'setup.py' with the following parameters 'build_ext --inplace'.

##### Possible run scenarios:
- you can run by specifying image name and number of samples
- or by running from preexisting 'dump.pickle' file

Possible run scenarios are commented inside of 'main.py' file, optimized cython code is located in 'optimized.pyx'. Input folder contains input files, output folder contains output files. Program generates 'output.png' from input.

Voronoi clipping is done as suggested by the following stack overflow answer.
In short, it duplicates and flips the input points around clipping lines. More can be found here: 
https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells/33602171#33602171

The code also uses a different method for calculating weights for color calculating, as it produces same (visually) image as with the algorithm described in 'Seminars.pdf' section 2.4, reference 3. It was used as it is faster and produces very small differences.
It calculates it by getting the size area stolen from neighbour polygon and then getting the percentage (area stolen from poly 1)/(all area stolen) and uses this as a weight.

Cython was used to help speed up some parts of the code(mainly computational) and other parts that seem to have benefited from it. It also uses memoization to prevent the same thing from being calculated multiple times.

This code does not use CUDA or any other parallel implementation, just CPU based implementation. The code is based on 2.4 section in 'Seminars.pdf'.

No help will be provided with the inclusion of this code. None. You can use as you wish, but I will not help you add/change/fix anything. You are on your own.

That's it folks!
