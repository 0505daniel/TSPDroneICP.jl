using TSPDrone

const MAX_DRONE_RANGE = Inf

mutable struct TSPDroneResult 
    total_cost::Float64
    truck_route::Vector{Int}
    drone_route::Vector{Int}
    Ct::Matrix{Float64}
    Cd::Matrix{Float64}
    flying_range::Float64
end

include("TSPDChain.jl")
include("rings.jl")
include("chainlets.jl")
include("tsp.jl")
include("utils.jl")
include("icp.jl")
# include("neuro.jl")
include("internals.jl")

const PROBLEM_TYPES = Dict(
    :TSPD => nothing, # Basic TSPD
)

const CHAIN_INITIALIZATION_METHODS = Dict(
    :Concorde => concorde_with_dummy_nodes,
    :LKH    => lkh_with_dummy_nodes,
    :FI       => fi_with_dummy_nodes,
    :NN       => nn_with_dummy_nodes,
    :CI       => ci_with_dummy_nodes,
    :Random   => random_with_dummy_nodes,
)

const CHAINLET_INITIALIZATION_METHODS = Dict(
    :Concorde => concorde_fixed_end,
    :LKH    => lkh_fixed_end,
    :FI       => fi_fixed_end,
    :NN       => nn_fixed_end,
    :CI       => ci_fixed_end,
    :Random   => random_fixed_end,
)

const CHAINLET_SOLVING_METHODS = Dict(
    :TSP_EP_all => TSPDrone.tsp_ep_all, 
)

const EVALUATION_METHODS = Dict(
    :Default => _evaluate_chainlet,
    # :Neuro   => _batch_evaluate_chainlet,
)

const SEARCH_METHODS = Dict(
    :Greedy   => _greedy_search_chainlet,
    :Roulette => _roulette_wheel_search_chainlet,
    :Softmax  => _softmax_search_chainlet,
)

function solve_tspd(
    truck_cost_mtx::Matrix{Float64},
    drone_cost_mtx::Matrix{Float64};
    problem_type::Symbol=:TSPD,
    flying_range::Float64=MAX_DRONE_RANGE, 
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy
)

    @assert size(truck_cost_mtx) == size(drone_cost_mtx)

    # the size of truck_cost_mtx and drone_cost_mtx is (n_customers + 1) x (n_customers + 1)
    # the first row and column are the depot
    # the rest of the rows and columns are the customers

    T, D = cost_matrices_with_dummy(truck_cost_mtx, drone_cost_mtx)

    result_chain = solve_tspd_by_ICP(
        T, 
        D; 
        flying_range=flying_range, 
        max_nodes=max_nodes, 
        chain_initialization_method=chain_initialization_method,
        chainlet_initialization_method=chainlet_initialization_method, 
        chainlet_solving_method=chainlet_solving_method,
        chainlet_evaluation_method=chainlet_evaluation_method,
        search_method=search_method
    )

    # Validation
    validate_tspd_chain(result_chain)

    return TSPDroneResult(result_chain.objective_value, result_chain.truck_route, result_chain.drone_route, result_chain.truck_cost_matrix, result_chain.drone_cost_matrix, result_chain.flying_range)

end

function solve_tspd(
    x::Vector{Float64}, 
    y::Vector{Float64}, 
    truck_cost_factor::Float64, 
    drone_cost_factor::Float64;
    problem_type::Symbol=:TSPD, 
    flying_range::Float64=MAX_DRONE_RANGE, 
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy
)

    # first node is the depot
    T, D = cost_matrices_with_dummy(x, y, truck_cost_factor, drone_cost_factor)

    result_chain = solve_tspd_by_ICP(
        T, 
        D; 
        flying_range=flying_range, 
        max_nodes=max_nodes, 
        chain_initialization_method=chain_initialization_method,
        chainlet_initialization_method=chainlet_initialization_method, 
        chainlet_solving_method=chainlet_solving_method,
        chainlet_evaluation_method=chainlet_evaluation_method,
        search_method=search_method
    )

    # Validation
    validate_tspd_chain(result_chain)

    return TSPDroneResult(result_chain.objective_value, result_chain.truck_route, result_chain.drone_route, result_chain.truck_cost_matrix, result_chain.drone_cost_matrix, result_chain.flying_range)

end 