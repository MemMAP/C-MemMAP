# C-MemMAP

This repo contains code accompaning the manuscript, "C-MemMAP: Clustering-driven Compact, Adaptable, and Generalizable Meta-LSTM Models for Memory Access Prediction"

## Dataset 
The trace uses the PARSEC benchmark(https://parsec.cs.princeton.edu/), generated using Pin tool, see example *Memory Reference Trace* (https://software.intel.com/sites/landingpage/pintool/docs/97503/Pin/html/)

## Specialized Model


## Delegated Model
### Running
```python3 ./Delegated_Model_Clustering.py```

The script uses the DCLSTM models in folder *Specialized_rerun_model*. The model weights of the DCLSTM models are concatenated, dimension reduced using PCA, and clustered using k-means. 