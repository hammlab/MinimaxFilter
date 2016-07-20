# MinimaxFilter
Minimax filter for private machine learning.
Based on 

### Abstract

### Main file descriptions

1. filterAlg.py 
2. learningAlg.py
3. minimaxFilter.py
4. kiwiel.py

filterAlg class object takes original data (X) and outputs
dimensionality-reduced version (G).
Currently, there are linear and NN-type filters. Look at the member
functions. They have to implement dg/du, where u is the parameter
vector (such as weights) of the filter.
My NN implementation is rudimentary. We should probably use a better
neural net library for large-scale experiments. By the way, the
library has to compute the Jacobian of the NN output w.r.t. the
parameters.

learningAlg class object takes G as input and outputs either the
desired prediction or the subject identity, depending on which labels
(y) it is trained on.
The class requires several member methods to be implemented, such as
df/dv, dfdu, etc.
Currently, only the softmax (=multclass logistic regression) is implemented.

minimaxFilter is the module that formulates my problem into a standard
minimax problem. Try running minimaxFilter.selftest1() and
selftest2(). By the way, the definition of f_util and f_priv here is
slightly different from the aistats paper.

kiwiel is the module that solves the standard minimax problem min_u
max_v f(u,v).
I've been wondering if I can use a simpler method, alternating min and
max, or alternating gradient descents.
The best resource for Kiwiel's method is: "Algorithms for Worst-case
Design and Applications
to Risk Management"

