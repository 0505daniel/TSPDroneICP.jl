# TSPDroneICP.jl

This package solves the Traveling Salesman Problem with Drone (TSP-D) with 1 truck and 1 drone. This implements the Iterative Chainlet Partitioning (ICP) algorithm and it's neural acceleration as proposed in the following paper:

* [Bogyrbaeyeva A., T. Yoon, H. Ko, S. Lim, H. Yun, and C. Kwon, A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone, working paper](https://arxiv.org/abs/2112.12545). 

If you use the ICP algorithm, please cite:
```
@misc{bogyrbayeva2021deep,
  title={A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone}, 
  author={Aigerim Bogyrbayeva and Taehyun Yoon and Hanbum Ko and Sungbin Lim and Hyokun Yun and Changhyun Kwon},
  year={2021},
  eprint={2112.12545},
  archivePrefix={arXiv},
  primaryClass={math.OC}
}
```
# License

This TSPDroneICP.jl package is in MIT License. However, the underlying Concorde solver is available for free only for academic research as described in the [Concorde](http://www.math.uwaterloo.ca/tsp/concorde.html) website.


# Requirements

```julia
] add https://github.com/chkwon/TSPDrone.jl
```
If you want just the ICP algorithm, the above will be sufficient. Skip the rest.

If you also want to utilize neural acceleration, first, activate `include("neuro.jl")` (Line 9) and `:Neuro   => _batch_evaluate_chainlet,` (Line 40) of `main.jl` .

Next, set up your `PyTorch` and `PyG` installations.
For example:
```console
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
```
Finally, in Julia:
```julia
julia> ENV["PYTHON"] = "your/python/path"
```
Then
```julia
julia> import Pkg; Pkg.build("PyCall")
```
Test if everything works fine:
```julia
julia> using TSPDroneICP
julia> TSPDroneICP.test_nicp()
```
which should not generate errors.
If it does not work properly, check if your Julia is connected with a proper Python installation. 
For example:
```julia
julia> using PyCall
julia> PyCall.python
"/your/python/path"
julia> PyCall.pyversion
v"your.python.version"
```
If it does not use the Python installation you like, try the above process again.

# Installation
To install:
```julia
import Pkg; Pkg.add(url="https://github.com/0505daniel/TSPDroneICP.jl")
```
or
```julia
] add https://github.com/0505daniel/TSPDroneICP.jl
```

# Using the Iterative Chainlet Partitioning (ICP) Algorithm

You can provide `x` and `y` coordinates of customers. 
The depot should be the first element in `x` and `y`.

Two parameters `truck_cost_factor` and `drone_cost_factor` will be multiplied to the Euclidean distance calculated from the coordinates. 
```julia 
using TSPDroneICP
n = 30 
x = rand(n); y = rand(n);
truck_cost_factor = 1.0 
drone_cost_factor = 0.5
result = solve_tspd(x, y, truck_cost_factor, drone_cost_factor)
@show result.total_cost;
@show result.truck_route;
@show result.drone_route;
```
returns
```
result.total_cost = 2.9097369509817126
result.truck_route = [1, 9, 11, 13, 21, 6, 20, 29, 8, 3, 7, 12, 24, 22, 25, 16, 19, 26, 10, 4, 31]
result.drone_route = [1, 2, 9, 15, 11, 18, 13, 14, 20, 23, 7, 27, 24, 17, 16, 28, 26, 5, 10, 30, 31]
```
where node `31` represents the depot as the final destination. 

You can also provide the cost matrices of truck and drone directly.
Again, the depot is labeled as `1`.
```julia
using TSPDroneICP
n = 30 
dist_mtx = rand(n, n)
dist_mtx = dist_mtx + dist_mtx' # symmetric distance only
truck_cost_mtx = dist_mtx .* 1.0
drone_cost_mtx = truck_cost_mtx .* 0.5 
result = solve_tspd(truck_cost_mtx, drone_cost_mtx)
@assert size(truck_cost_mtx) == size(drone_cost_mtx) == (n, n)
@show result.total_cost
@show result.truck_route
@show result.drone_route
```
returns
```
result.total_cost = 6.321744019456498
result.truck_route = [1, 9, 29, 6, 4, 12, 28, 27, 13, 24, 5, 17, 20, 7, 25, 19, 26, 22, 18, 11, 31]
result.drone_route = [1, 3, 9, 29, 6, 23, 12, 16, 27, 10, 5, 15, 7, 8, 25, 2, 26, 14, 18, 30, 11, 21, 31]
```
where again node `31` represets the depot as the final destination.

## Summary of the Result
Use `print_summary(result)`:
```julia
julia> print_summary(result)
Ring #1:
  - Truck        = 0.17988883875173492 : [1, 3]
  - Drone        = 0.11900891950265155 : [1, 4, 3]
  - Length       = 0.17988883875173492
Ring #2:
  - Truck        = 0.4784476248243221 : [3, 9]
  - Drone        = 0.27587675362585756 : [3, 7, 9]
  - Length       = 0.4784476248243221
Ring #3:
  - Truck        = 0.445749847855226 : [9, 6]
  - Drone        = 0.48831605249544785 : [9, 10, 6]
  - Length       = 0.48831605249544785
Ring #4:
  - Truck        = 0.9269158918021541 : [6, 5, 8, 11]
  - Drone        = 0.8714473929102112 : [6, 2, 11]
  - Length       = 0.9269158918021541
Total Cost = 2.073568407873659
```
## Options for ICP
Optional keyword arguments for `solve_tspd`:
```julia
problem_type::Symbol=:TSPD,
flying_range::Float64=MAX_DRONE_RANGE, 
max_nodes::Int=20, 
chain_initialization_method::Symbol=:Concorde,
chainlet_initialization_method::Symbol=:FI, 
chainlet_solving_method::Symbol=:TSP_EP_all,
chainlet_evaluation_method::Symbol=:Default,
search_method::Symbol=:Greedy
```

- ``problem_type``: The type of problem to solve. Currently, the only supported problem type is ``:TSPD``, which is the basic TSP-D. This flag is reserved for future extensions.

- ``flying_range``: The limited flying range of the drone. The default value is `Inf`. The flying range is compared with the values in the drone cost matrix; that is, `drone_cost_mtx` or the Euclidean distance multiplied by `drone_cost_factor`. 

- ``max_nodes``: The maximum number of nodes allowed for each chainlet.

- ``chain_initialization_method``: The method to construct initial TSP tour for the chain can be one of the following:
  - `:Concorde` : [Concorde TSP Solver](http://www.math.uwaterloo.ca/tsp/concorde/index.html)
  - `:LKH` : [LKH heuristic solver](http://webhotel4.ruc.dk/~keld/research/LKH/)
  - `:FI` : Farthest insertion 
  - `:NN` : Nearest neighbor 
  - `:CI` : Cheapest insertion
  - `:Random` : Random 
  > **Note:** If you're using ``:LKH``, the underlying LKH library comes in a different license. Check with the [LKH library homepage](http://webhotel4.ruc.dk/~keld/research/LKH-3/).

- ``chainlet_initialization_method``: The method to construct initial tour for chainlets. Options are same with ``chain_initialization_method``.

- ``chainlet_solving_method``: The method for optimizing each chainlet. Currently, only ``:TSP_EP_all`` is implemented (see [Agatz et al. 2018](https://doi.org/10.1287/trsc.2017.0791)).

- ``chainlet_evaluation_method``: Specifies how to evaluate chainlet improvements. When set to ``:Default``, the method calculates the actual improvement. To use neural acceleration, set this option to ``:Neuro``, which bypasses the direct execution of TSP-EP-all.

- ``search_method``: The strategy for selecting a chainlet to update. Options:
  - ``:Greedy``: Selects the chainlet with the highest improvement.
  - ``:Roulette``: Uses roulette wheel selection.
  - ``:Softmax``: Uses a softmax probability distribution.
  > **Note:** When using neural acceleration, the selection strategy will be ``:Greedy`` regardless of what option is selected.
  

# Related Projects
- [TSPDrone.jl](https://github.com/chkwon/TSPDrone.jl): Julia implementation of [Optimization Approaches for the Traveling Salesman Problem with Drone](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2017.0791?journalCode=trsc_) and [A Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drone](https://www.sciencedirect.com/science/article/abs/pii/S0968090X22003941).
- [Concorde.jl](https://github.com/chkwon/Concorde.jl): Julia wrapper of the [Concorde TSP Solver](http://www.math.uwaterloo.ca/tsp/concorde/index.html).
- [LKH.jl](https://github.com/chkwon/LKH.jl): Julia wrapper of the [LKH heuristic solver](http://webhotel4.ruc.dk/~keld/research/LKH/).
- [TravelingSalesmanHeuristics.jl](https://github.com/evanfields/TravelingSalesmanHeuristics.jl): Julia package for simple traveling salesman problem heuristics. 
