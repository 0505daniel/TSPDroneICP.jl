using LKH
using Concorde
using TravelingSalesmanHeuristics

"""
Functions for TSP tour construction
"""

# Construct the initial TSP tour using Concorde
function concorde_with_dummy_nodes(T::Matrix{Float64})::Vector{Int}

    N = size(T)[1] - 1
    tsp_tour, _ = Concorde.solve_tsp(round.(Int, T[1:end-1, 1:end-1] .* 1000))
    push!(tsp_tour, N+1) # adding the final depot

    # Ensure the second route node is less than the last route node (For algorithm to behave deterministically)
    if tsp_tour[2] > tsp_tour[end-1]
        tsp_tour = [tsp_tour[1]; reverse(tsp_tour[2:end-1]); tsp_tour[end]]
    end

    return tsp_tour
end

# Construct the initial TSP tour using LKH-3
function lkh_with_dummy_nodes(T::Matrix{Float64})::Vector{Int}

    N = size(T)[1] - 1
    tsp_tour, _ = LKH.solve_tsp(round.(Int, T[1:end-1, 1:end-1] .* 1000))
    push!(tsp_tour, N+1) # adding the final depot

    return tsp_tour
end

# Construct the initial TSP tour using Farthest Insertion
function fi_with_dummy_nodes(T::Matrix{Float64})::Vector{Int}

    N = size(T)[1] - 1
    tsp_tour, _ = TravelingSalesmanHeuristics.farthest_insertion(T[1:end-1, 1:end-1]; firstcity = 1, do2opt = false)
    # tsp_tour, _ = TravelingSalesmanHeuristics.farthest_insertion(T[1:end-1, 1:end-1]; firstcity = 1, do2opt = true)
    tsp_tour[end] = N+1 # Convert the final depot to dummy node

    return tsp_tour
end

# Construct the initial TSP tour using Nearest Neighbor
function nn_with_dummy_nodes(T::Matrix{Float64})::Vector{Int}

    N = size(T)[1] - 1
    tsp_tour, _ = TravelingSalesmanHeuristics.nearest_neighbor(T[1:end-1, 1:end-1]; firstcity = 1, do2opt = false)
    tsp_tour[end] = N + 1 # Convert the final depot to dummy node

    return tsp_tour
end

# Construct the initial TSP tour using Cheapest Insertion
function ci_with_dummy_nodes(T::Matrix{Float64})::Vector{Int}

    N = size(T)[1] - 1
    tsp_tour, _ = TravelingSalesmanHeuristics.cheapest_insertion(T[1:end-1, 1:end-1]; firstcity = 1, do2opt = false)
    tsp_tour[end] = N + 1 # Convert the final depot to dummy node

    return tsp_tour
end

# Construct the initial TSP tour using Random Insertion
function random_with_dummy_nodes(T::Matrix{Float64})::Vector{Int}

    N = size(T)[1] - 1
    tsp_tour = [1; shuffle(2:N)]
    tsp_cost = sum(T[tsp_tour[i], tsp_tour[i+1]] for i in 1:N-1) + T[tsp_tour[end], tsp_tour[1]]
    push!(tsp_tour, N+1)

    return tsp_tour
end

# Construct the initial TSP tour with fixed end

#NOTE: TOO SLOW
function concorde_fixed_end(chain::Vector{Int}, dis::Matrix{Float64})::Vector{Int}
    ls = length(chain)
    dis_adj = Matrix{Float64}(undef, ls, ls) 
    @inbounds for i in 1:ls, j in 1:ls
        dis_adj[i, j] = dis[i, j]
    end
    @inbounds for i in 2:(ls-1)
        dis_adj[ls, i] = 1000.0
        dis_adj[i, ls] = 1000.0 # Check why this is needed
    end
    dis_adj[ls, 1] = 0.0
    dis_adj[1, ls] = 0.0 # Check why this is needed
    route, _ = Concorde.solve_tsp(round.(Int, dis_adj .* 1000)) # Solve TSP with Concorde.solve_tsp
    # @show route

    if route[2] == ls # In case for the reversed path
        route = [route[1]; reverse(route[2:end])]
    end

    # @show route[end]

    @assert ls == route[end] "The last node of the tour must be the same as the last node of the input list."

    return route
end

function lkh_fixed_end(chain::Vector{Int}, dis::Matrix{Float64})::Vector{Int}
    ls = length(chain)
    dis_adj = zeros(ls, ls)
    @inbounds for i in 1:ls, j in 1:ls
        dis_adj[i, j] = dis[i, j]
    end
    @inbounds for i in 2:(ls-1)
        dis_adj[ls, i] = 1000.0
        dis_adj[i, ls] = 1000.0 # Check why this is needed
    end
    dis_adj[ls, 1] = 0.0
    dis_adj[1, ls] = 0.0 # Check why this is needed
    route, _ = LKH.solve_tsp(round.(Int, dis_adj .* 1000)) # Solve TSP with LKH.solve_tsp
    # @show route

    if route[2] == ls # In case for the reversed path
        route = [route[1]; reverse(route[2:end])]
    end

    # @show route[end]

    @assert ls == route[end] "The last node of the tour must be the same as the last node of the input list."

    return route
end


function fi_fixed_end(chain::Vector{Int}, dis::Matrix{Float64})::Vector{Int}
    ls = length(chain)
    @assert length(ls) < 2 "There must be at least two nodes for a valid TSP."
    
    remaining = Set(2:ls-1)
    tour = [1, ls]

    while !isempty(remaining)

        # Selection step
        farthest, farthest_dist = 0, -1.0
        @inbounds for city in remaining
            max_dist_to_tour = maximum([dis[city, tour_city] for tour_city in tour])
            if max_dist_to_tour > farthest_dist
                farthest, farthest_dist = city, max_dist_to_tour
            end
        end
        
        # Insertion step
        best_insert_pos, best_insert_cost = 0, Inf
        @inbounds for i in 1:(length(tour)-1)
            if tour[i] != 0
                cost_increase = dis[tour[i], farthest] + dis[farthest, tour[i+1]] - dis[tour[i], tour[i+1]]
                if cost_increase < best_insert_cost
                    best_insert_pos, best_insert_cost = i, cost_increase
                end
            end
        end
        
        insert!(tour, best_insert_pos+1, farthest)
        delete!(remaining, farthest)
    end
    
    return tour
end

function ci_fixed_end(chain::Vector{Int}, dis::Matrix{Float64})::Vector{Int}
    ls = length(chain)
    @assert ls >= 2 "There must be at least two nodes for a valid TSP."
    
    remaining = Set(2:ls-1)
    tour = [1, ls]

    while !isempty(remaining)
        # Selection step: Nearest Insertion
        nearest, nearest_dist = 0, Inf
        @inbounds for city in remaining
            min_dist_to_tour = minimum([dis[city, tour_city] for tour_city in tour])
            if min_dist_to_tour < nearest_dist
                nearest, nearest_dist = city, min_dist_to_tour
            end
        end
        
        # Insertion step: Insert nearest city at the best position
        best_insert_pos, best_insert_cost = 0, Inf
        @inbounds for i in 1:(length(tour)-1)
            cost_increase = dis[tour[i], nearest] + dis[nearest, tour[i+1]] - dis[tour[i], tour[i+1]]
            if cost_increase < best_insert_cost
                best_insert_pos, best_insert_cost = i, cost_increase
            end
        end
        
        insert!(tour, best_insert_pos+1, nearest)
        delete!(remaining, nearest)
    end
    
    return tour
end

function nn_fixed_end(chain::Vector{Int}, dis::Matrix{Float64})::Vector{Int}
    ls = length(chain)
    @assert ls >= 2 "There must be at least two nodes for a valid TSP."
    
    remaining = Set(2:ls-1)
    tour = [1]

    while !isempty(remaining)
        # Selection step: Greedy Insertion
        best_city, best_cost = 0, Inf
        @inbounds for city in remaining
            @inbounds for i in 1:length(tour)
                cost_increase = dis[tour[i], city]
                if cost_increase < best_cost
                    best_city, best_cost = city, cost_increase
                end
            end
        end
        
        # Insertion step: Append best city to the tour
        push!(tour, best_city)
        delete!(remaining, best_city)
    end
    
    # Append the end node
    push!(tour, ls)
    
    return tour
end

function random_fixed_end(chain::Vector{Int}, dis::Matrix{Float64})::Vector{Int}
    ls = length(chain)
    @assert ls >= 2 "There must be at least two nodes for a valid TSP."
    
    remaining = Set(2:ls-1)
    tour = [1]

    while !isempty(remaining)
        # Selection step: Random Insertion
        random_city = rand(collect(remaining))
        
        # Insertion step: Append random city to the tour
        push!(tour, random_city)
        delete!(remaining, random_city)
    end
    
    # Append the end node
    push!(tour, ls)
    
    return tour
end