
using PyCall
project_root = realpath(joinpath(@__DIR__, ".."))
py_dir = joinpath(project_root, "nn_fr")
config_path = joinpath(py_dir, "trained", "model_config.json")

py"""
import os
import sys; sys.path.append($py_dir);

import torch
import pickle
import json

# from net import TSPDGraphTransformerNetwork
# from infer import get_graph, predict_chainlet_length, batch_prediction, remove_module_prefix

# # Load configuration from JSON file
# config_file_path = os.path.join("../nn/trained", "model_config.json")
# with open(config_file_path, "r") as f:
#     configs = json.load(f)

# model_name = "neural_cost_predictor"
# model_config = configs["models"][model_name]
# model_path = model_config["model_path"]
# config = model_config["config"]

# print("CUDA Available:", torch.cuda.is_available())
# print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = TSPDGraphTransformerNetwork(
#     in_channels=config['pos_encoding_dim'],
#     hidden_channels=config['pos_encoding_dim'],
#     out_channels=config['pos_encoding_dim'],
#     heads=config['heads'],
#     beta=config['beta'],
#     dropout=config['dropout'],
#     normalization=config['normalization'],
#     num_gat_layers=config['num_gat_layers'],
#     activation=config['activation']
# )

# # Load the state_dict
# state_dict = torch.load(model_path, map_location=device, weights_only=True)

# # Remove the 'module.' prefix if it exists
# state_dict = remove_module_prefix(state_dict)

# # Load the state_dict into the model
# model.load_state_dict(state_dict)

# model = model.to(device)
# model.eval()

from net_fr import TSPDGraphTransformerNetworkFlyingRange
from infer_fr import get_graph, predict_chainlet_length, batch_prediction, remove_module_prefix

config_file_path = $config_path
config_dir = os.path.dirname(config_file_path)

with open(config_file_path, "r") as f:
    configs = json.load(f)

model_name = "neural_cost_predictor_fr"
model_config = configs["models"][model_name]
model_path = os.path.join(config_dir, model_config["model_path"])
config = model_config["config"]

print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dict = torch.load(model_path, map_location=device, weights_only=True)
state_dict = state_dict['model_state_dict'] # Extract the correct dictionary
state_dict = remove_module_prefix(state_dict)

model = TSPDGraphTransformerNetworkFlyingRange(dropout=0.0, edge_dim=3, **config)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
"""

function _batch_evaluate_chainlet(Chain::TSPDChain)
    Chain.chainlet_truck_routes = [Vector{Int}() for _ in 1:Chain.chainlet_sequence_length]
    Chain.chainlet_drone_routes = [Vector{Int}() for _ in 1:Chain.chainlet_sequence_length]
    chainlets_to_predict = []
    chainlets_to_predict_scale = []
    indices_to_update = []

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

            _, _, scale = min_max_scale_matrix(truck_cost_submatrix)

            # push!(chainlets_to_predict, Dict("initial_route" => initial_route, "C_t" => truck_cost_submatrix, "C_d" => drone_cost_submatrix))
            push!(chainlets_to_predict, Dict("initial_route" => initial_route, "C_t" => truck_cost_submatrix, "C_d" => drone_cost_submatrix, "flying_range" => Chain.flying_range))
            push!(chainlets_to_predict_scale, scale)
            push!(indices_to_update, i)
        end
    end

    # Batch prediction for all unseen chainlets
    if !isempty(chainlets_to_predict)
        # pred_chainlet_lengths = py"batch_prediction"(py"device", py"model", py"config['pos_encoding_dim']", chainlets_to_predict, chainlets_to_predict_scale)
        pred_chainlet_lengths = py"batch_prediction"(py"device", py"model", 8, chainlets_to_predict, chainlets_to_predict_scale)
    
        @inbounds for j in eachindex(indices_to_update)
            i = indices_to_update[j]
            chainlet = Chain.chainlets[i]
            chainlet_tuple::Tuple{Vararg{Int}} = tuple(chainlet...)

            pred_chainlet_length = pred_chainlet_lengths[j]
            Chain.chainlet_increment[i] = Chain.chainlet_costs[i] - pred_chainlet_length
            Chain.chainlet_truck_routes[i] = chainlets_to_predict[j]["initial_route"]
            # Store opt_chainlet_length directly (not chainlet_increment) to avoid issues when chainlet_costs changes
            Chain.chainlet_increments[chainlet_tuple] = (pred_chainlet_length, Chain.chainlet_truck_routes[i], Chain.chainlet_drone_routes[i])
        end
    end
        
end

function _neuro_search_chainlet(Chain::TSPDChain)::Bool
    
    all_seen = false
    
    while !all_seen

        target = argmax(Chain.chainlet_increment)
        flag = !isempty(Chain.chainlet_drone_routes[target]) # Check if tsp-ep-all has been run for the chainlet
        target_chainlet = Chain.chainlets[target]
        target_chainlet_tuple::Tuple{Vararg{Int}} = tuple(target_chainlet...)

        if flag
            # Cache stores (opt_chainlet_length, tr_idx, dr_idx)
            opt_chainlet_length_cached = Chain.chainlet_increments[target_chainlet_tuple][1]
            Chain.chainlet_truck_routes[target] = Chain.chainlet_increments[target_chainlet_tuple][2]
            Chain.chainlet_drone_routes[target] = Chain.chainlet_increments[target_chainlet_tuple][3]
            # Compute chainlet_increment from cached opt_chainlet_length and current chainlet_costs
            Chain.chainlet_increment[target] = Chain.chainlet_costs[target] - opt_chainlet_length_cached

        else
            truck_cost_submatrix::Matrix{Float64} = Chain.truck_cost_matrix[target_chainlet, :][:, target_chainlet]
            drone_cost_submatrix::Matrix{Float64} = Chain.drone_cost_matrix[target_chainlet, :][:, target_chainlet]

            opt_chainlet_length, tr_idx, dr_idx = isnothing(Chain.flying_range) ? Chain.chainlet_solving_function(truck_cost_submatrix, drone_cost_submatrix, Chain.chainlet_truck_routes[target]) : Chain.chainlet_solving_function(truck_cost_submatrix, drone_cost_submatrix, Chain.chainlet_truck_routes[target], flying_range=Chain.flying_range)
                
            new_chainlet_inc = Chain.chainlet_costs[target] - opt_chainlet_length
   
            Chain.chainlet_increment[target] = Chain.chainlet_costs[target] - opt_chainlet_length
            Chain.chainlet_truck_routes[target] = tr_idx
            Chain.chainlet_drone_routes[target] = dr_idx
            # Store opt_chainlet_length directly (not chainlet_increment) to avoid issues when chainlet_costs changes
            Chain.chainlet_increments[target_chainlet_tuple] = (opt_chainlet_length, Chain.chainlet_truck_routes[target], Chain.chainlet_drone_routes[target])

            new_chainlet = chainlet_creation(target_chainlet, tr_idx, dr_idx)
   
            new_chainlet_tuple::Tuple{Vararg{Int}} = tuple(new_chainlet...)
            # For result chainlet, store opt_chainlet_length (same as the optimized original chainlet)
            Chain.chainlet_increments[new_chainlet_tuple] = (opt_chainlet_length, tr_idx, dr_idx)
        end

        if Chain.chainlet_increment[target] > 0.01
            _iterate_only_rings(Chain, target)
            return all_seen
        else 
            Chain.chainlet_increment[target] = - Inf

            # Check if all chainlets have been seen
            all_seen = all(!isempty, Chain.chainlet_drone_routes)
        end
    end

    return all_seen

end

function test_nicp()
    
    include("src/neuro.jl")

    n = 100
    dist_mtx = rand(n, n)
    dist_mtx = dist_mtx + dist_mtx' # symmetric distance only
    truck_cost_mtx = dist_mtx .* 1.0
    drone_cost_mtx = truck_cost_mtx .* 0.5 
    @assert size(truck_cost_mtx) == size(drone_cost_mtx) == (n, n)
    
    max_dist = maximum(truck_cost_mtx)
    flying_range_percentage = rand(5:50)
    flying_range_percentage = Float64(flying_range_percentage)

    flying_range = max_dist * flying_range_percentage / 200.0

    result = solve_tspd(truck_cost_mtx, drone_cost_mtx; chainlet_evaluation_method=:Neuro, flying_range=flying_range)
    @info "Testing n = $n / Neuro ICP (NICP)"

    # TSPDroneICP.print_summary(result)
    @show result.total_cost
    @show result.truck_route
    @show result.drone_route

    @info "Successfully tested NICP"
end