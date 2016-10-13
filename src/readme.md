#### File descriptions


[filterAlg.py](filterAlg.py) class takes original data (X) and outputs dimensionality-reduced version (G).
Currently, there are linear (filterAlg_Linear.py) and NN-type filters (filterAlg_NN.py). 
Look at the member functions. They have to implement 'dg/du', where 'u' is the parameter vector (such as weights) of the filter.
My NN implementation is rudimentary, and should be replaced with anoother neural net library for large-scale experiments.
By the way, the library has to compute the Jacobian of the NN output w.r.t. the parameters.

[learningAlg.py](learningAlg.py) class takes G as input and outputs either the target prediction (z) or sensitive prediction (y), depending on which label it is trained on.
The class requires several member methods to be implemented, such as 'df/dv', 'dfdu', etc.
Currently, only the softmax (=multclass logistic regression) is implemented.

[minimaxFilter](minimaxFilter.py) is a module that formulates the privacy-utility optimization problem into a min-diff-max  problem. Try running 'minimaxFilter.selftest1()' and 'minimaxFilter.selftest2()'. 

[kiwiel](kiwiel.py) is a module that solves the standard minimax problem 'min_u max_v f(u,v).' Descriptions are given in [Hamm'15].

[alternatingOptim](alternatingOptim.py) also solves the standard minimax problem using the alternating algorithm proposed in [Hamm'16].

[privacyLDA](privacyLDA.py) is a heuristic proposed in [Hamm'16] which is similar to linear discriminant analysis (LDA).

[privacyPLS](privacyPLS.py) is an implementation of another privacy-preserving filter from Enev et al., Annual Computer Security Applications Conference (ACSAC), 2012.
