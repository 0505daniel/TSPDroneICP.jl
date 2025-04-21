
function _calculate_ring_costs(truck_rings::Vector{Vector{Int}}, drone_rings::Vector{Vector{Int}}, truck_cost_matrix::Matrix{Float64}, drone_cost_matrix::Matrix{Float64})::Vector{Float64}
    
    num_rings = length(truck_rings)
    ring_costs = zeros(Float64, num_rings)
    
    k = 1
    @inbounds for i in 1:num_rings
        truck_ring = truck_rings[i]
        drone_ring = drone_rings[i]
        
        truck_cost = travel_cost(truck_ring, truck_cost_matrix)
        drone_cost = travel_cost(drone_ring, drone_cost_matrix)
        
        ring_costs[k] = max(truck_cost, drone_cost)
        k += 1
    end

    return ring_costs
end


function _divide_into_rings(truck_route::Vector{Int}, drone_route::Vector{Int}, truck_cost_matrix::Matrix{Float64}, drone_cost_matrix::Matrix{Float64}; end_chainlet::Bool=false)
    combined_nodes = intersect(truck_route, drone_route)
    rings = [[id] for id in combined_nodes if id in truck_route]

    truck_rings = [copy(ring) for ring in rings]
    drone_rings = [copy(ring) for ring in rings]

    _update_rings(combined_nodes, drone_rings, drone_route)
    _update_rings(combined_nodes, truck_rings, truck_route)

    _update_rings(combined_nodes, rings, drone_route)
    _update_rings(combined_nodes, rings, truck_route)

    !end_chainlet ? pop!(rings) : nothing

    pop!(truck_rings)
    pop!(drone_rings)

    @inbounds for i in eachindex(truck_rings)
        push!(truck_rings[i], combined_nodes[i+1])
        push!(drone_rings[i], combined_nodes[i+1])
    end

    ring_costs = _calculate_ring_costs(truck_rings, drone_rings, truck_cost_matrix, drone_cost_matrix)
    # objective_value = sum(ring_costs)

    return rings, ring_costs
end


function _update_rings(combined_nodes, rings, route)
    idx = 1
    @inbounds for k in eachindex(route)
        if route[k] != combined_nodes[idx+1]
            if route[k] != combined_nodes[idx]
                push!(rings[idx], route[k])
            end
        else
            idx += 1
        end
    end
end

# Construct near TSPD-TSP solution
function reconstruct_tsp_from_rings(rings)
    return vcat(rings...)  
end
