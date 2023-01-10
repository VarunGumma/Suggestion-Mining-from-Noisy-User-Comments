# Suggestion-Mining-from-Noisy-Data

Project work done at National University of Singapore (NUS) under the guidance of [Dr. Aditya Karanam](https://www.comp.nus.edu.sg/disa/bio/karanam/). 

The work includes detecting _Suggestion sentences_ and _Suggested features_ from noisy user comments. 

Checkpoints for the trained/fine-tuned models are available in the ```tar``` file, which contains two folders, ```saved_weights [src_feat]``` and ```saved_weights [src_sug]```. Copy each of these to the respective directories. Make sure to remove the ```[src_feat]``` and ```[src_sug]``` from the names after copying.

It is recommended to run all the code on a good GPU, preferrably with the ```amp_backend``` set as [apex](https://github.com/NVIDIA/apex) for more efficient computations. In case ```apex``` is not installed in your system, you remove the line ```amp_backend="apex"``` from the ```pl.Trainer``` class constructor.
