
function is_subgroup(chainlet1::Vector{Int}, chainlet2::Vector{Int})::Bool
    len1 = length(chainlet1)
    len2 = length(chainlet2)
    if len1 > len2
        return false
    end
    for i in 1:(len2 - len1 + 1)
        if chainlet2[i:(i + len1 - 1)] == chainlet1
            return true
        end
    end
    return false
end

function generate_chainlets(Chain::TSPDChain)
    Chain.chainlets = Vector{Int}[] 
    Chain.chainlet_sizes = Int[]
    Chain.chainlet_locations = Int[]   

    node_limit = Chain.max_nodes - 1 # Technical issue since Chain.rings does not contain the end node

    i = 1 # i represents the seqence length of chainlets
    while i <= length(Chain.rings)
        chainlet = Vector{Int}()
        num_nodes = 0
        j = 0 # j represents the sizes of the chainlet

        flag = false # flag to check if at least one loop is added to the chainlet
        # while num_nodes < Chain.max_nodes && (i+j) < length(Chain.rings)
        #     next_ring_length = length(Chain.rings[i + j])
        #     if num_nodes + next_ring_length > Chain.max_nodes && flag
        #         break
        #     end
        while num_nodes < node_limit && (i+j) < length(Chain.rings)
            next_ring_length = length(Chain.rings[i + j])
            if num_nodes + next_ring_length > node_limit && flag
                break
            end
            append!(chainlet, Chain.rings[i+j])
            flag = true
            num_nodes += next_ring_length
            j += 1
        end
        
        if !isempty(Chain.chainlets) && Chain.rings[i+j][1] == Chain.chainlets[end][end]
            i += 1
            continue
        else
            push!(Chain.chainlet_locations, i)
            push!(Chain.chainlet_sizes, j)
            append!(chainlet, Chain.rings[i+j][1])
            push!(Chain.chainlets, chainlet)
    
            if (i + j) == length(Chain.rings)
                break
            else
                i += 1
            end
        end
    end

    Chain.chainlet_sequence_length = length(Chain.chainlets)
    Chain.chainlet_costs = zeros(Chain.chainlet_sequence_length)

    @inbounds for i in 1:Chain.chainlet_sequence_length
        chainlet = Chain.chainlets[i]
        Chain.chainlet_costs[i] = sum(Chain.ring_costs[k] for k in Chain.chainlet_locations[i]:(Chain.chainlet_locations[i] + Chain.chainlet_sizes[i] - 1))
    end

    Chain.chainlet_increment = zeros(Chain.chainlet_sequence_length)

    # for chainlet in Chain.chainlets
    #     if length(chainlet) > Chain.max_nodes
    #         @show chainlet
    #     end
    #     @assert length(chainlet) <= Chain.max_nodes "Chainlet size is greater than the maximum number of nodes"
    # end
end

function _evaluate_chainlet(Chain::TSPDChain)
    Chain.chainlet_truck_routes = [Vector{Int}() for _ in 1:Chain.chainlet_sequence_length]
    Chain.chainlet_drone_routes = [Vector{Int}() for _ in 1:Chain.chainlet_sequence_length]
    @inbounds for i in 1:Chain.chainlet_sequence_length
        chainlet = Chain.chainlets[i]
        chainlet_tuple::Tuple{Vararg{Int}} = tuple(chainlet...)

        if chainlet_tuple in keys(Chain.chainlet_increments)
            # Cache stores (opt_chainlet_length, tr_idx, dr_idx)
            opt_chainlet_length_cached = Chain.chainlet_increments[chainlet_tuple][1]
            Chain.chainlet_truck_routes[i] = Chain.chainlet_increments[chainlet_tuple][2]
            Chain.chainlet_drone_routes[i] = Chain.chainlet_increments[chainlet_tuple][3]
            # Compute chainlet_increment from cached opt_chainlet_length and current chainlet_costs
            Chain.chainlet_increment[i] = Chain.chainlet_costs[i] - opt_chainlet_length_cached
        else
            truck_cost_submatrix::Matrix{Float64} = Chain.truck_cost_matrix[chainlet, :][:, chainlet]
            drone_cost_submatrix::Matrix{Float64} = Chain.drone_cost_matrix[chainlet, :][:, chainlet]
            initial_route = Chain.chainlet_initialization_function(chainlet, truck_cost_submatrix) 
            
            opt_chainlet_length, tr_idx, dr_idx = isnothing(Chain.flying_range) ? Chain.chainlet_solving_function(truck_cost_submatrix, drone_cost_submatrix, initial_route) : Chain.chainlet_solving_function(truck_cost_submatrix, drone_cost_submatrix, initial_route, flying_range=Chain.flying_range)

            Chain.chainlet_increment[i] = Chain.chainlet_costs[i] - opt_chainlet_length
            Chain.chainlet_truck_routes[i] = tr_idx
            Chain.chainlet_drone_routes[i] = dr_idx
            
            # Store opt_chainlet_length directly (not chainlet_increment) to avoid issues when chainlet_costs changes
            Chain.chainlet_increments[chainlet_tuple] = (opt_chainlet_length, tr_idx, dr_idx)

            # Also cache the result chainlet
            new_chainlet = chainlet_creation(chainlet, tr_idx, dr_idx)
            new_chainlet_tuple::Tuple{Vararg{Int}} = tuple(new_chainlet...)
            # For result chainlet, opt_chainlet_length equals chainlet_costs (no improvement), so store chainlet_costs
            Chain.chainlet_increments[new_chainlet_tuple] = (Chain.chainlet_costs[i], tr_idx, dr_idx)    
        end
    end    
end

function chainlet_creation(chainlet::Vector{Int}, truck_route_idx::Vector{Int}, drone_route_idx::Vector{Int})::Vector{Int}
    tr = [chainlet[i] for i in truck_route_idx]
    dr = [chainlet[i] for i in drone_route_idx]
    cmb = intersect(tr, dr)

    new_chainlet = zeros(Int, length(chainlet))
    tr_idx, dr_idx, idx = 1, 1, 1
    while tr_idx <= length(tr)
        current_tr = tr[tr_idx]
        new_chainlet[idx] = current_tr
        tr_idx += 1
        idx += 1

        if current_tr in cmb
            dr_idx = findfirst(isequal(current_tr), dr) + 1
            while dr_idx <= length(dr) && !(dr[dr_idx] in cmb)
                new_chainlet[idx] = dr[dr_idx]
                idx += 1
                dr_idx += 1
            end
        end
    end

    return new_chainlet
end

