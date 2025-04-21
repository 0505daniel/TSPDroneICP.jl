# Empty function to be used as a placeholder for unimplemented functions
function empty_function(::Any...)
    @warn "Called an unimplemented function"
    return nothing
end

function get_dimension_of_square_matrix(A::Matrix)::Int
    n, _ = size(A)
    return n
end

function cost_matrices_with_dummy(truck_cost_mtx::Matrix{Float64}, drone_cost_mtx::Matrix{Float64})::Tuple{Matrix{Float64}, Matrix{Float64}}
    Ct::Matrix{Float64} = [
        truck_cost_mtx          truck_cost_mtx[:, 1];
        truck_cost_mtx[1, :]'    0.0
    ]

    Cd::Matrix{Float64} = [
        drone_cost_mtx          drone_cost_mtx[:, 1];
        drone_cost_mtx[1, :]'    0.0
    ]

    return Ct, Cd
end

function _cost_matrices_with_dummy(x, y, speed_of_truck, speed_of_drone)
    n_nodes = length(x)
    @assert length(x) == length(y)

    dist = zeros(Float64, n_nodes, n_nodes)
    @inbounds for i in 1:n_nodes
        @inbounds for j in 1:n_nodes
            dist[i, j] = (x[i]-x[j])^2 + (y[i]-y[j])^2 |> sqrt 
        end
    end

    Ct = speed_of_truck .* dist 
    Cd = speed_of_drone .* dist 

    return Ct, Cd
end

function cost_matrices_with_dummy(x, y, speed_of_truck, speed_of_drone)
    xx = copy(x)
    yy = copy(y)
    push!(xx, x[1])
    push!(yy, y[1])
    return _cost_matrices_with_dummy(xx, yy, speed_of_truck, speed_of_drone)
end


function calculate_duration_matrices(tspeed::Float64, dspeed::Float64, depot::Vector{Float64}, customers::Matrix{Float64})
    # This function is used when the input are the node coordinates and the speed of vehicles.
    # It takes the input and returns the matrices that include travel times for each vehicle between nodes
    num_of_nodes = size(customers)[1]
    D = Matrix{Float64}(undef, num_of_nodes + 2, num_of_nodes + 2)
    T = Matrix{Float64}(undef, num_of_nodes + 2, num_of_nodes + 2)
    D[1, 1] = 0.0
    D[num_of_nodes+2, num_of_nodes+2] = 0.0
    D[1, num_of_nodes+2] = 0.0
    D[num_of_nodes+2, 1] = 0.0
    T[1, 1] = 0.0
    T[num_of_nodes+2, num_of_nodes+2] = 0.0
    T[1, num_of_nodes+2] = 0.0
    T[num_of_nodes+2, 1] = 0.0
    @inbounds for i in 1:num_of_nodes
        D[i+1, i+1] = 0.0
        T[i+1, i+1] = 0.0
        D[1, i+1] = euclidean(depot, customers[i, :]) / dspeed
        D[i+1, 1] = D[1, i+1]
        D[num_of_nodes+2, i+1] = D[1, i+1]
        D[i+1, num_of_nodes+2] = D[1, i+1]
        T[1, i+1] = euclidean(depot, customers[i, :]) / tspeed
        # T[1, i+1] = cityblock(depot, customers[i, :]) / tspeed
        T[i+1, 1] = T[1, i+1]
        T[num_of_nodes+2, i+1] = T[1, i+1]
        T[i+1, num_of_nodes+2] = T[1, i+1]
        @inbounds for j in 1:num_of_nodes
            D[i+1, j+1] = euclidean(customers[i, :], customers[j, :]) / dspeed
            D[j+1, i+1] = D[i+1, j+1]
            T[i+1, j+1] = euclidean(customers[i, :], customers[j, :]) / tspeed
            # T[i+1, j+1] = cityblock(customers[i, :], customers[j, :]) / tspeed
            T[j+1, i+1] = T[i+1, j+1]
        end
    end

    return T, D
end

function min_max_scale(values)
    min_val = minimum(values)
    max_val = maximum(values)
    scaled_values = (values .- min_val) ./ (max_val - min_val)
    return scaled_values, min_val, max_val
end

function min_max_scale_matrix(matrix)
    min_val = minimum(matrix)
    max_val = maximum(matrix)
    return (matrix .- min_val) ./ (max_val - min_val), min_val, max_val
end

function travel_cost(path::Vector{Int}, C::Matrix{T}) where T
    # @show path
    sum = zero(T)
    @inbounds for i in 1:length(path)-1
        sum += C[path[i], path[i+1]]
    end
    return sum
end

# TSPDrone.jl objective value calculation function
function objective_value(truck_route, drone_route, Ct, Cd)
    combined_nodes = intersect(truck_route, drone_route)
    obj_val = 0.0
    @inbounds for i in 1:length(combined_nodes)-1
        j1 = combined_nodes[i]
        j2 = combined_nodes[i+1]
        
        t_idx1 = findfirst(x -> x == j1, truck_route)
        t_idx2 = findfirst(x -> x == j2, truck_route)
        t_cost = travel_cost(truck_route[t_idx1:t_idx2], Ct)

        d_idx1 = findfirst(x -> x == j1, drone_route)
        d_idx2 = findfirst(x -> x == j2, drone_route)
        d_cost = travel_cost(drone_route[d_idx1:d_idx2], Cd)

        obj_val += max(t_cost, d_cost)
    end
    return obj_val
end

# Function validating the initial TSP tour
function validate_intial_tsp(tour::Vector{Int}, N::Int)
    @assert length(tour) == N+1 "Tour must have length N+1."
    @assert tour[1] == 1 "Tour must start with 1."
    @assert tour[end] == N+1 "Tour must end with N+1."
    @assert sort(tour[2:end-1]) == collect(2:N) "Tour must contain 1..N in any order, plus the final depot."
end

# Function validating the output tour of the TSPD
function validate_tspd_chain(Chain::TSPDChain)

    # Combine truck and drone routes to check all visited nodes
    visited_nodes = vcat(Chain.truck_route, Chain.drone_route)

    # Remove duplicate nodes (if any)
    unique_nodes = unique(visited_nodes)

    # Expected nodes from 1 to the number of nodes (assuming truck_cost_matrix defines the total nodes)
    expected_nodes = Set(1:size(Chain.truck_cost_matrix, 1))

    # Validate that all required nodes are visited exactly once
    @assert Set(unique_nodes) == expected_nodes "Not all nodes are visited in the tour"

    cmb = intersect(Chain.truck_route, Chain.drone_route)   

    # Check if drone visits at most one node between combined nodes
    @inbounds for i in 1:length(cmb)-1
        # Find the indices of the combined nodes in the drone route
        idx1 = findfirst(==(cmb[i]), Chain.drone_route)
        idx2 = findfirst(==(cmb[i+1]), Chain.drone_route)
        
        # If there's more than one node between the combined nodes in the drone route, throw an error
        @assert idx2 - idx1 <= 2 "Drone visits more than one node between combined nodes"
    end

    # Check if the flying range is feasible when there is a drone node
    @inbounds for i in 1:length(Chain.drone_route)-1
        # If the current node is a drone node
        if !(Chain.drone_route[i] in cmb)
            # Find the next combined node in the drone route
            next_combined_node_index = findnext(in(cmb), Chain.drone_route, i+1)
        
            # Calculate the total distance from the truck to the drone node and back to the truck
            total_distance = Chain.drone_cost_matrix[Chain.drone_route[i-1], Chain.drone_route[i]] + Chain.drone_cost_matrix[Chain.drone_route[i], Chain.drone_route[next_combined_node_index]]
            
            # If the total distance is greater than the flying range, throw an error
            @assert total_distance <= Chain.flying_range "Flying range is not feasible for drone node $(Chain.drone_route[i])"
        end
    end

    obj_val = objective_value(Chain.truck_route, Chain.drone_route, Chain.truck_cost_matrix, Chain.drone_cost_matrix)
    @assert Chain.objective_value ≈ obj_val atol=1e-5 "Objective value mismatch. Calculated: $obj_val, Expected: $(Chain.objective_value)"
end
