### Minimax Filter
#### is a machine learning approach to preserve privacy against inference attacks
---
Moitvation

![concept figure](minimaxfilter.jpg | width=100)

The library allows devices (Android, iOS, and python clients) to learn a common classifier/regression model with differential privacy, by solving the distributed ERM problem: min_w f(w) = 1/M sum_{i=1}^M f_i(w), where f_i(w) = 1/n sum_j l(h_w(x_{ij}), y_{ij}).
The library implements private distributed synchronous risk minimization based on [**Hamm'15**], using [Google Firebase](https://firebase.google.com/) as a simple and robust syncrhonization method.  This idea was featured in [Gigaom] (https://gigaom.com/2015/01/22/researchers-show-a-machine-learning-network-for-connected-devices/).

Choosing the type and amount of noise to guarantee differential privacy is left to the library user; the type and the amount 
depend on model assumptions. Please see [Chaudhuri'11], [Rajkumar'12], [Song'13], [Bassily'14], [Hamm'16].
If noise is not used, this library can also serve as a crowd-based, parallel/distributed optimization framework [Tsitsiklis'84], [Agarwal'11], [Dekel'11]. 

### Getting Started
---
#### 1. Download files in /src and /test
#### 2. Run [test/test_NN_genki.py](test/test_NN_genki.py) to test a two-layer sigmoid NN network with softmax output layers on Genki dataset.
The Genki datat is ...
Results.
#### 3. Run [test/test_all_genki.py](test/test_all_genki.py)
Results.


### Description of src files
---
See [src/readme.md](src/readme.md) for the summary of source files


### References
---
* [Hamm'15]: J. Hamm, "Preserving privacy of continuous high-dimensional data with minimax filters." 
In Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics (AISTATS), 2015.
* [Hamm'16a]: J. Hamm, "Enhancing utility and privacy with noisy minimax filters." Under review, 2016.
*[Hamm'16b]: J. Hamm, "Mimimax Filter: A Learning Approach to Preserve Privacy from Inference Attacks." arXiv, 2016


### License
---
Released under the Apache License 2.0.  See the [LICENSE.txt](LICENSE.txt) file for further details.





