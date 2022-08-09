# Suggestion-Mining-from-Noisy-Data

Project work done at National University of Singapore (NUS) under the guidance of [Dr. Aditya Karanam](https://www.comp.nus.edu.sg/disa/bio/karanam/). 

The work includes detecting _Suggestion sentences_ and _Suggested features_ from noisy user comments. 

Saved weights for all the models that were trained can be found [here](https://drive.google.com/file/d/1neyQX8_AVOACauWsZ9BC9gL8GGB49w3p/view?usp=sharing). The zip file contains two folders, ```saved_weights [src_feat]``` and ```saved_weights [src_sug]```. Copy each of these to the respective folders.

Make sure to remove the ```[src_feat]``` and ```[src_sug]``` from the names after copying.

It is recommended to run all the code on a good GPU, preferrably with the ```amp_backend``` set as [apex](https://github.com/NVIDIA/apex) for more efficient computations.