# Single-Pass Tensor Decomposition with Sketching

## Background

Given limited storage resources, many higher order datasets such as video, PDE, and weather benefit from being modeled as a streaming case, where data is in the form of linear updates. Even with these linear updates, storing sketches of the data may be needed to combat the problem of limited storage. This paper proposes a new low-rank tensor approximation sketch algorithm that only passes through the original tensor once during calculation. This algorithm also provides a theoretical approximation guarantee, as well as computational speed comparable to existing non-streaming algorithms. Simulations as well as experiments on real weather and PDE data show the efficiency of this algorithm. 

## Objective
- Minimize the communication cost by reducing the access to the original large tensor into a single pass 
- Enable a stream of input data stored in a distributed setting 

## Prerequisite

We implemented our algorithms in python3. Please first install the dependency packages by running the following commands in the terminal. 
```
pip install numpy
pip install matplotlib 
pip install scipy
pip install sklearn 
pip install tensorflow 
pip install -U tensorly
```

## Results 

Please refer to [simulation.ipynb] for simulation on synthetic data and [simulation_weather.ipynb] for simulation on real-world weather data. 


## Local Package installation 
```
pip install -e .
```