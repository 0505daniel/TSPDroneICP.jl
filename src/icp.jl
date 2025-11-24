
# code warntype clear
function _greedy_search_chainlet(Chain::TSPDChain)
    target = argmax(Chain.chainlet_increment)
    _iterate_only_rings(Chain, target)
    return nothing
end

function _roulette_wheel_search_chainlet(Chain::TSPDChain)
    positive_indices = findall(x -> x > 0, Chain.chainlet_increment)
    total = sum(Chain.chainlet_increment[positive_indices])

    rand_num = rand() * total
    running_sum = 0.0

    @inbounds for index in positive_indices
        running_sum += Chain.chainlet_increment[index]

        if running_sum > rand_num
            _iterate_only_rings(Chain, index)
            return nothing
        end
    end
    return nothing
end

function _softmax_search_chainlet(Chain::TSPDChain)

    # # Only consider the chainlets with increment greater than 0.01
    # filtered_indices = findall(x -> x > 0.01, Chain.chainlet_increment)
    # exp_increments = exp.(Chain.chainlet_increment[filtered_indices])
    # probabilities = exp_increments / sum(exp_increments)

    exp_increments = exp.(Chain.chainlet_increment)
    probabilities = exp_increments / sum(exp_increments)

    cumulative_probabilities = cumsum(probabilities)

    rand_num = rand()
    index = findfirst(x -> x >= rand_num, cumulative_probabilities)
    if index !== nothing
        _iterate_only_rings(Chain, index)
    end

    return nothing
end


function _iterate_only_rings(Chain::TSPDChain, target::Int)

    updated_route = Chain.chainlets[target]
    updated_truck_route = updated_route[Chain.chainlet_truck_routes[target]]
    updated_drone_route = updated_route[Chain.chainlet_drone_routes[target]]

    update_start_node = Chain.chainlets[target][1]
    update_end_node = Chain.chainlets[target][end]

    # Update truck and drone route
    splice!(Chain.truck_route, findfirst(x -> x == update_start_node, Chain.truck_route):findfirst(x -> x == update_end_node, Chain.truck_route), updated_truck_route)
    splice!(Chain.drone_route, findfirst(x -> x == update_start_node, Chain.drone_route):findfirst(x -> x == update_end_node, Chain.drone_route), updated_drone_route)

    N::Int = size(Chain.truck_cost_matrix, 1)
    end_chainlet::Bool = Chain.chainlets[target][end] == N

    updated_rings, updated_ring_costs = _divide_into_rings(updated_truck_route, updated_drone_route, Chain.truck_cost_matrix, Chain.drone_cost_matrix; end_chainlet=end_chainlet)

    default::UnitRange{Int} = (Chain.chainlet_locations[target]:(Chain.chainlet_locations[target] + Chain.chainlet_sizes[target] - 1)) 
    range::UnitRange{Int} = !end_chainlet ? 
    default : (Chain.chainlet_locations[target]:(Chain.chainlet_locations[target] + Chain.chainlet_sizes[target]))

    # Update rings
    splice!(Chain.rings, range, updated_rings); splice!(Chain.ring_costs, default, updated_ring_costs); Chain.objective_value = sum(Chain.ring_costs)

    Chain.iteration += 1

end


function run_ICP(T::Matrix{Float64}, D::Matrix{Float64};
    problem_type::Symbol=:TSPD, 
    flying_range::Float64=Inf, 
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy,
    tsp_tour::Union{Vector{Int}, Nothing}=nothing,
    truck_route::Union{Vector{Int}, Nothing}=nothing,
    drone_route::Union{Vector{Int}, Nothing}=nothing)

    n1, n2 = size(T)
    n_nodes = n1 - 1
    @assert size(T) == size(D)
    
    # Validate that truck_route and drone_route are provided together if either is provided
    if (truck_route !== nothing) != (drone_route !== nothing)
        error("Both truck_route and drone_route must be provided together, or neither should be provided")
    end
    
    # If truck_route and drone_route are provided, use them directly (skip TSP generation and exact_partitioning)
    if truck_route !== nothing && drone_route !== nothing
        # Use provided routes directly, skip TSP generation and exact_partitioning
    # If only tsp_tour is provided, use it for exact_partitioning (skip TSP generation)
    elseif tsp_tour !== nothing
        total_cost, truck_route, drone_route = TSPDrone.exact_partitioning(tsp_tour, T, D; flying_range=flying_range)
    # Normal flow: generate TSP tour and run exact_partitioning
    else
        @assert haskey(CHAIN_INITIALIZATION_METHODS, chain_initialization_method) "Invalid chain initialization method: $chain_initialization_method"
        chain_initialization_function::Function = CHAIN_INITIALIZATION_METHODS[chain_initialization_method]
        tsp_tour = chain_initialization_function(T)
        total_cost, truck_route, drone_route = TSPDrone.exact_partitioning(tsp_tour, T, D; flying_range=flying_range)
    end
    
    chain = TSPDChain(truck_route, drone_route, T, D; 
    problem_type=problem_type,
    flying_range=flying_range, 
    max_nodes=max_nodes,
    chain_initialization_method=chain_initialization_method, 
    chainlet_initialization_method=chainlet_initialization_method, 
    chainlet_solving_method=chainlet_solving_method,
    chainlet_evaluation_method=chainlet_evaluation_method,
    search_method=search_method)

    chain.rings, chain.ring_costs = _divide_into_rings(chain.truck_route, chain.drone_route, chain.truck_cost_matrix, chain.drone_cost_matrix; end_chainlet=true)
    chain.objective_value = sum(chain.ring_costs)
    
    generate_chainlets(chain)
    chain.evaluation_function(chain)

    all_seen::Bool = chainlet_evaluation_method == :Neuro ? false : true # Flag for NICP
    previous_objective_value::Float64 = chain.objective_value
    
    while (maximum(chain.chainlet_increment)) >= 0.01 || !all_seen

        previous_iteration::Int = chain.iteration
        
        if chainlet_evaluation_method == :Neuro
            all_seen = chain.search_function(chain)
        else
            chain.search_function(chain)
        end

        generate_chainlets(chain)
        chain.evaluation_function(chain)

        # Check for false iteration due to numerical error accumulation of TSP-EP-all
        # If objective_value didn't improve (or got worse) AND an iteration occurred, terminate.
        # Only check if an iteration actually occurred (was incremented)
        if chain.iteration > previous_iteration && chain.objective_value >= previous_objective_value
            @info "Algorithm Manually Terminated: No improvement detected (false iteration due to numerical error)"
            chain.iteration -= 1
            @info "Iteration count decremented"
            break
        end
        previous_objective_value = chain.objective_value
        
        # println("Algorithm Terminated")
    end

    return chain
end


