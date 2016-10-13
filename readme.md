### Minimax Filter
#### is a machine learning approach to preserve privacy against inference attacks
---

![concept figure](minimaxfilter2.jpg "Example minimax filter")

#### Abstract

Preserving privacy of continuous and/or high-dimensional data such as images, videos
and audios, can be challenging with syntactic anonymization methods such as k-anonymity
which are designed for discrete attributes. Differential privacy, which provides a different
and more formal type of privacy, has shown more success in sanitizing continuous data.
However, both syntactic and differential privacy are susceptible to inference attacks, i.e., an
adversary can accurately guess sensitive attributes from insensitive attributes. This paper
proposes a learning approach to finding a minimax filter of raw features which retains infor-
mation for target tasks but removes information from which an adversary can infer sensitive
attributes. Privacy and utility of filtered data are measured by expected risks, and an opti-
mal tradeoff of the two goals is found by a variant of minimax optimization. Generalization
performance of the empirical solution is analyzed and and a new and simple optimization
algorithm is presented. In addition to introducing minimax filter, the paper proposes noisy
minimax filter that combines minimax filter and differentially private noisy mechanism,
and compare resilience to inference attack and differentially privacy both quantitatively
and qualitatively. Experiments with several real-world tasks including facial expression
recognition, speech emotion recognition, and activity recognition from motion, show that
the minimax filter can simultaneously achieve similar or better target task accuracy and
lower inference accuracy, often significantly lower, than previous methods.


### Getting Started
---
#### 1. Download files in /src and /test
Make sure you can access scripts in /src, for example by downloading files in both /src and /test in the same folder.
Description of the files are in [src/readme.md](src/readme.md) and [test/readme.md](test/readme.md).
The Genki dataset [test/genki.mat](test/genki.mat) was originally downloaded from http://mplab.ucsd.edu. 

#### 2. Run [test/test_NN_genki.py](test/test_NN_genki.py) 
The task is to learn a filter of face images from the Genki dataset which allows accurate classification of smile vs non-smile but prevents accurate classification of male vs female. 

The script finds a minimax filter by alternating optimization. The filer is a two-layer sigmoid neural net and the classifiers are softmax classifiers. 

The script will run for a few minutes on a desktop. 
After 50 iterations, the filter will achieve ~88% accuracy in facial expression classification and ~66% accuracy in gender classification.
```
minimax-NN: rho=10.000000, d=10, trial=0, rate1=0.88, rate2=0.66
```
Results will be save to a file named 'test_NN_genki.npz'

#### 3. Run [test/test_all_genki.py](test/test_all_genki.py)
The task is the same as before (accurate facial expression and inaccurate gender classification.)

The script trains several private and non-private algorithms for the same task, including a linear minimax filter . on the same data. finds a min-diff-max optimal filter by alternating optimization. The filer is a two-layer sigmoid neural net and the classifiers are softmax classifiers. 

The script will also run for a few minutes on a desktop. 
Below is an example result from the script. The rate1 is the accuracy of expression classification and the rate 2 is the accuracy of gender classification.
```
rand: d=10, trial=0, rate1=0.705000, rate2=0.705000

pca: d=10, trial=0, rate1=0.840000, rate2=0.665000

pls: d=10, trial=0, rate1=0.850000, rate2=0.685000

alt: rho=10.000000, d=10, trial=0, rate1=0.825000, rate2=0.520000
```


### References
---
* [Hamm'15]: J. Hamm, "Preserving privacy of continuous high-dimensional data with minimax filters." 
In Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics (AISTATS), 2015.
* [Hamm'16b]: J. Hamm, "Mimimax Filter: A Learning Approach to Preserve Privacy from Inference Attacks." arXiv:1610.03577, 2016


### License
---
Released under the Apache License 2.0.  See the [LICENSE.txt](LICENSE.txt) file for further details.





