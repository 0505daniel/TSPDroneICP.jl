const MAX_DRONE_RANGE = Inf
const MAX_TIME_LIMIT = Inf
function tspd_objective_value(truck_route, drone_route, Ct, Cd)
    combined_nodes = intersect(truck_route, drone_route)
    obj_val = 0.0
    for i in 1:length(combined_nodes)-1
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

function precompute_dist_order(Cd)
    n, _ = size(Cd)
    dist_order = repeat(reshape(1:n, 1, :), n, 1)
    for (i, row) in enumerate(eachrow(dist_order))
        row .= sort(collect(row), by = x -> Cd[i, x])
    end
    return dist_order
end
function exact_partitioning_boosted(initial_tour, Ct, Cd; flying_range=MAX_DRONE_RANGE, dist_order=nothing)
    n, _ = size(Ct)
    r = initial_tour
    T = fill(Inf, n, n)
    M = fill(-1, n, n)
    sum_Ct = zeros(n, n)
    for i in 1:n-1
        sum_Ct[i, i+1] = Ct[r[i], r[i+1]]
        for j in i+2:n
            sum_Ct[i, j] = sum_Ct[i, j-1] + Ct[r[j-1], r[j]]
        end
    end
    for i in 1:n-1
        T[i, i+1] = sum_Ct[i, i+1]
    end
    inv_r = zeros(Int, n)
    for i in 1:n
        inv_r[r[i]] = i
    end
    if flying_range == MAX_DRONE_RANGE
        @inbounds for i in 1:n-2
            @inbounds for k in i+1:n-1
                for j = k+1:n
                    t_cost = sum_Ct[i, k-1] + Ct[r[k-1], r[k+1]] + sum_Ct[k+1, j]
                    d_cost = Cd[r[i], r[k]] + Cd[r[k], r[j]]
                    cost = max(t_cost, d_cost)
                    if T[i, j] > cost
                        T[i, j] = cost
                        M[r[i], r[j]] = r[k]
                    end
                    if t_cost > d_cost
                        break
                    end
                end
            end
        end
    else
        if isnothing(dist_order)
        	dist_order = precompute_dist_order(Cd)
        end
        J = Vector{Int}(undef, n)
        for k_tour in 2:n-1
            k_node = r[k_tour]
            @simd for j in k_tour:n
                J[j] = j+1
            end
            for i_node in dist_order[k_node, :]
                i_tour = inv_r[i_node]
                if i_tour >= k_tour
                    continue
                end
                j_tour = J[k_tour]
                prev_j_tour = k_tour
                while j_tour !== n+1
                    j_node = r[j_tour]
                    d_cost = Cd[i_node, k_node] + Cd[k_node, j_node]
                    if d_cost > flying_range
                        j_tour = J[j_tour]
                        J[prev_j_tour] = j_tour
                        continue
                    end
                    t_cost = sum_Ct[i_tour, k_tour-1] + Ct[r[k_tour-1], r[k_tour+1]] + sum_Ct[k_tour+1, j_tour]
                    cost = max(t_cost, d_cost)
                    if T[i_tour, j_tour] > cost
                        T[i_tour, j_tour] = cost
                        M[i_node, j_node] = k_node
                    end
                    if t_cost > d_cost
                        break
                    end
                    prev_j_tour = j_tour
                    j_tour = J[j_tour]
                end
            end
        end
    end
    V = zeros(n)
    P = fill(-1, n)
    V[1] = 0
    @inbounds for i in 2:n
        VV = [V[k] + T[k, i] for k in 1:i-1]
        am = argmin(VV)
        V[i] = VV[am]
        P[i] = r[am]
    end
    # Retrieving solutions.
    combined_nodes = Int[]
    current_idx = n
    current = r[current_idx]
    while current != -1
        push!(combined_nodes, current)
        current_idx = inv_r[current]
        current = P[current_idx]
    end
    reverse!(combined_nodes)
    drone_only_nodes = Int[]
    drone_route = Int[]
    push!(drone_route, combined_nodes[1])
    @assert combined_nodes[1] == r[1]
    @inbounds for i in 1:length(combined_nodes)-1
        j1 = combined_nodes[i]
        j2 = combined_nodes[i+1]
        if M[j1, j2] != -1
            push!(drone_only_nodes, M[j1, j2])
            push!(drone_route, M[j1, j2])
        end
        push!(drone_route, j2)
    end
    truck_route = setdiff(initial_tour, drone_only_nodes)
    final_time = V[end]
    # obj_val = tspd_objective_value(truck_route, drone_route, Ct, Cd)
    # @assert isapprox(obj_val, final_time)
    return final_time, truck_route, drone_route
end
function exact_partitioning(initial_tour, Ct, Cd; flying_range=MAX_DRONE_RANGE)
    n, _ = size(Ct)
    r = initial_tour
    T = fill(Inf, n, n)
    M = fill(-1, n, n)
    sum_Ct = zeros(n, n)
    for i in 1:n-1
        sum_Ct[i, i+1] = Ct[r[i], r[i+1]]
        for j in i+2:n
            sum_Ct[i, j] = sum_Ct[i, j-1] + Ct[r[j-1], r[j]]
        end
    end
    for i in 1:n-1
        T[i, i+1] = sum_Ct[i, i+1]
    end
    @inbounds for i in 1:n-1
        @inbounds for j in i+1:n
            @inbounds for k in i+1:j-1
                Tk1 = Cd[r[i], r[k]] + Cd[r[k], r[j]]
                if Tk1 <= flying_range
                    Tk2 = sum_Ct[i, k-1] + sum_Ct[k+1, j] + Ct[r[k-1], r[k+1]]
                    # if Tk2 <= flying_range
                    #     Tk = max(Tk1, Tk2)
                    #     if Tk < T[i, j]
                    #         T[i, j] = Tk
                    #         M[r[i], r[j]] = r[k]
                    #     end
                    # end
                    Tk = max(Tk1, Tk2)
                    if Tk < T[i, j]
                        T[i, j] = Tk
                        M[r[i], r[j]] = r[k]
                    end
                end
            end
        end
    end
    V = zeros(n)
    P = fill(-1, n)
    V[1] = 0
    @inbounds for i in 2:n
        VV = [V[k] + T[k, i] for k in 1:i-1]
        am = argmin(VV)
        V[i] = VV[am]
        P[i] = r[am]
    end
    combined_nodes = Int[]
    current_idx = n
    current = r[current_idx]
    inv_r = zeros(Int, n)
    for i in 1:n
        inv_r[r[i]] = i
    end
    while current != -1
        push!(combined_nodes, current)
        current_idx = inv_r[current]
        current = P[current_idx]
    end
    reverse!(combined_nodes)
    drone_only_nodes = Int[]
    drone_route = Int[]
    push!(drone_route, combined_nodes[1])
    @assert combined_nodes[1] == r[1]
    @inbounds for i in 1:length(combined_nodes)-1
        j1 = combined_nodes[i]
        j2 = combined_nodes[i+1]
        if M[j1, j2] != -1
            push!(drone_only_nodes, M[j1, j2])
            push!(drone_route, M[j1, j2])
        end
        push!(drone_route, j2)
    end
    truck_route = setdiff(initial_tour, drone_only_nodes)
    final_time = V[end]
    # obj_val = tspd_objective_value(truck_route, drone_route, Ct, Cd)
    # @assert isapprox(obj_val, final_time)
    return final_time, truck_route, drone_route
end


function two_point_move(tour, i, j)
    if i >= j
        return tour, false
    end
    tmp = copy(tour)
    tmp[j] = tour[i]
    tmp[i] = tour[j]
    return tmp, true
end
function one_point_move(tour, i, j)
    tmp = copy(tour)
    deleteat!(tmp, i)
    insert!(tmp, j, tour[i])
    return tmp, true
end
function two_opt_move(tour, i, j)
    if i >= j
        return tour, false
    end
    tmp = copy(tour)
    tmp[i:j] = tour[j:-1:i]
    return tmp, true
end


function tsp_ep_all(
    Ct,
    Cd,
    init_tour;
    local_search_methods=[two_point_move, one_point_move, two_opt_move],
    flying_range=MAX_DRONE_RANGE,
    time_limit=MAX_TIME_LIMIT,
    ep_boost=true
)
    n, _ = size(Ct)
    improved = true
    if ep_boost
    	# dist_order = precompute_dist_order(Cd)
		# best_obj, t_route, d_route = exact_partitioning_boosted(init_tour, Ct, Cd, flying_range=flying_range, dist_order = dist_order)
        best_obj, t_route, d_route = exact_partitioning_boosted(init_tour, Ct, Cd, flying_range=flying_range)
    else
    	best_obj, t_route, d_route = exact_partitioning(init_tour, Ct, Cd, flying_range=flying_range)
    end
    best_tour = copy(init_tour)
    best_t_route = copy(t_route)
    best_d_route = copy(d_route)
    if isempty(local_search_methods)
        return best_obj, best_t_route, best_d_route
    end
    time0 = time()
    is_time_over = false
    while improved && !is_time_over
        improved = false
        cur_best_obj = best_obj
        cur_best_tour = copy(best_tour)
        cur_best_t_route = copy(best_t_route)
        cur_best_d_route = copy(best_d_route)
        for i in 2:n-1
            if is_time_over
                break
            end
            for local_search in local_search_methods
                for j in 2:n-1
                    if is_time_over
                        break
                    end
                    new_tour, is_valid = local_search(best_tour, i, j)
                    is_time_over = time() - time0 > time_limit
                    if is_valid
                    	if ep_boost
                    		# ep_time, t_route, d_route = exact_partitioning_boosted(new_tour, Ct, Cd, flying_range=flying_range, dist_order=dist_order)
                            ep_time, t_route, d_route = exact_partitioning_boosted(new_tour, Ct, Cd, flying_range=flying_range)
                    	else
                        	ep_time, t_route, d_route = exact_partitioning(new_tour, Ct, Cd, flying_range=flying_range)
                        end

                        if ep_time < cur_best_obj
                            cur_best_tour = copy(new_tour)
                            cur_best_t_route = copy(t_route)
                            cur_best_d_route = copy(d_route)
                            cur_best_obj = ep_time
                            improved = true
                        end
                    end
                end
            end
        end
        if improved
            best_obj = cur_best_obj
            best_tour = cur_best_tour
            best_t_route = cur_best_t_route
            best_d_route = cur_best_d_route
        end
    end
    return best_obj, best_t_route, best_d_route
end

function tsp_ep_all_original(
    Ct,
    Cd,
    init_tour;
    local_search_methods=[two_point_move, one_point_move, two_opt_move],
    flying_range=MAX_DRONE_RANGE,
    time_limit=MAX_TIME_LIMIT,
)
    return tsp_ep_all(Ct, Cd, init_tour, local_search_methods=local_search_methods, flying_range=flying_range, time_limit=time_limit, ep_boost=false)
end

function tsp_ep_all_boosted(
    Ct,
    Cd,
    init_tour;
    local_search_methods=[two_point_move, one_point_move, two_opt_move],
    flying_range=MAX_DRONE_RANGE,
    time_limit=MAX_TIME_LIMIT,
)
    return tsp_ep_all(Ct, Cd, init_tour, local_search_methods=local_search_methods, flying_range=flying_range, time_limit=time_limit, ep_boost=true)
end
