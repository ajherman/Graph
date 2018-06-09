# Graph

Author: TEA

Mission: Explore strange new colorings, seek out new independent sets and cliques, boldly prove facts about J(v,k,i) and related graphs.

Before jokes are approved for the master branch, they should be showcased here in dev first.

TO DO:

1) Make a separate file to store all the functions we are using regularly, and remove them from the files we actually run
2) Make a tensorflow version of some of the computationally heavy parts, to see if they run faster on GPU (more vectorized possible?)
2b) Yeah, like the iterations in the outer loop are independent of one another, so they should be able to run at the same time...
3) Make a script for generating arrays of independene numbers for various v,k,i values
4) Low hanging: Make a function that computes the fractional chromatic # /fractional clique number of a vertex transitive graph
5) This is a matter of taste, but I'd like to make our program write to a text file rather than printing...that way it's not super annoying if you accidentally close a window that had some results you just took 10 hours to compute
6) Put better comments in scripts, so that if we want to share this code with someone someday they won't tear their hair our trying to decipher it
