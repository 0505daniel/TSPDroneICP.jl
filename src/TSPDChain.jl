
mutable struct TSPDChain

    # Required
    truck_route::Vector{Int} 
    drone_route::Vector{Int} 
    truck_cost_matrix::Matrix{Float64}
    drone_cost_matrix::Matrix{Float64}

    # Optional
    problem_type::Symbol
    flying_range::Float64
    max_nodes::Int
    chain_initialization_method::Symbol
    chainlet_initialization_method::Symbol
    chainlet_solving_method::Symbol
    chainlet_evaluation_method::Symbol
    search_method::Symbol

    # Internal

    # Rings & Chainlets
    rings::Vector{Vector{Int}}      
    ring_costs::Vector{Float64}
    chainlet_increment::Vector{Float64}
    chainlet_increments::Dict{Tuple{Vararg{Int}}, Tuple{Float64, Vector{Int}, Vector{Int}}} # Cash register
    chainlet_sequence_length::Int
    chainlet_costs::Vector{Float64}
    chainlets::Vector{Vector{Int}}
    chainlet_sizes::Vector{Int}
    chainlet_locations::Vector{Int}
    chainlet_truck_routes::Vector{Vector{Int}}
    chainlet_drone_routes::Vector{Vector{Int}}
    objective_value::Float64
    iteration::Int

    # Configurations
    chainlet_initialization_function::Function
    chainlet_solving_function::Function
    evaluation_function::Function
    search_function::Function
end


function TSPDChain(
    truck_route, drone_route, truck_cost_matrix, drone_cost_matrix;
    problem_type::Symbol=:TSPD, 
    flying_range=Inf, 
    max_nodes=20, 
    chain_initialization_method=:Concorde,
    chainlet_initialization_method=:FI, 
    chainlet_solving_method=:TSP_EP_all,
    chainlet_evaluation_method=:Default,
    search_method=:Greedy)

    # Configure
    @assert haskey(PROBLEM_TYPES, problem_type) "Invalid problem type: $problem_type"
    
    @assert haskey(CHAINLET_INITIALIZATION_METHODS, chainlet_initialization_method) "Invalid chainlet initialization method: $chainlet_initialization_method"
    chainlet_initialization_function = CHAINLET_INITIALIZATION_METHODS[chainlet_initialization_method]

    @assert haskey(CHAINLET_SOLVING_METHODS, chainlet_solving_method) "Invalid chainlet solving method: $chainlet_solving_method"
    chainlet_solving_function = CHAINLET_SOLVING_METHODS[chainlet_solving_method]

    @assert haskey(EVALUATION_METHODS, chainlet_evaluation_method) "Invalid chainlet evaluation method: $chainlet_evaluation_method"
    evaluation_function = EVALUATION_METHODS[chainlet_evaluation_method]

    @assert haskey(SEARCH_METHODS, search_method) "Invalid search method: $search_method"
    if chainlet_evaluation_method == :Neuro
        search_function = _neuro_search_chainlet
    else
        search_function = SEARCH_METHODS[search_method]
    end

    Chain = TSPDChain(
        # Required
        truck_route,
        drone_route,
        truck_cost_matrix,
        drone_cost_matrix,

        # Optional
        problem_type,
        flying_range,
        max_nodes,
        chain_initialization_method,
        chainlet_initialization_method,
        chainlet_solving_method,
        chainlet_evaluation_method,
        search_method,

        # Internal

        # Rings & Chainlets
        Vector{Vector{Int}}(), # rings
        Float64[], # ring_lengths
        Float64[], # chainlet_increment
        Dict{Tuple{Vararg{Int}}, Tuple{Float64, Vector{Int}, Vector{Int}}}(), # chainlet_increments
        0, # chainlet_sequence_length
        Float64[], # chainlet_costs
        Vector{Vector{Int}}(), # chainlets
        Vector{Int}(), # chainlet_sizes
        Vector{Int}(), # chainlet_locations
        Vector{Vector{Int}}(), # chainlet_truck_routes
        Vector{Vector{Int}}(), # chainlet_drone_routes
        0.0, # objective_value
        0, # iteration

        # Configurations
        chainlet_initialization_function,
        chainlet_solving_function,
        evaluation_function,
        search_function
    )

    return Chain
end