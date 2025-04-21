using DelimitedFiles
using Distances


function Calculate_distance_matrices_Agatz(alpha::Float64, depot::Tuple{Float64,Float64}, Nodes::Vector{Tuple{Float64,Float64}})
    num_of_nodes = length(Nodes)
    D = zeros(num_of_nodes + 1, num_of_nodes + 1)
    T = zeros(num_of_nodes + 1, num_of_nodes + 1)
    Dp = zeros(num_of_nodes)
    Tp = zeros(num_of_nodes)
    for i = 1:num_of_nodes
        Dp[i] = euclidean(depot, Nodes[i]) / alpha
        Tp[i] = euclidean(depot, Nodes[i])
        for j = 1:num_of_nodes
            D[i+1, j+1] = euclidean(Nodes[i], Nodes[j]) / alpha
            T[i+1, j+1] = euclidean(Nodes[i], Nodes[j])
        end
    end

    D[2:end, 1] = Dp
    D[1, 2:end] = Dp
    T[2:end, 1] = Tp
    T[1, 2:end] = Tp

    return T, D
end

function Calculate_distance_matrices_with_dummy_Agatz(alpha::Float64, depot::Tuple{Float64,Float64}, Nodes::Vector{Tuple{Float64,Float64}})
    num_of_nodes = length(Nodes)
    D = zeros(num_of_nodes, num_of_nodes)
    T = zeros(num_of_nodes, num_of_nodes)
    Dp = zeros(num_of_nodes)
    Tp = zeros(num_of_nodes)
    for i = 1:num_of_nodes
        Dp[i] = euclidean(depot, Nodes[i]) / alpha
        Tp[i] = euclidean(depot, Nodes[i])
        for j = 1:num_of_nodes
            D[i, j] = euclidean(Nodes[i], Nodes[j]) / alpha
            T[i, j] = euclidean(Nodes[i], Nodes[j])
        end
    end


    DD = zeros(num_of_nodes + 2, num_of_nodes + 2)
    TT = zeros(num_of_nodes + 2, num_of_nodes + 2)
    DD[2:num_of_nodes+1, 2:num_of_nodes+1] = D
    DD[2:num_of_nodes+1, 1] = Dp
    DD[1, 2:num_of_nodes+1] = Dp
    DD[2:num_of_nodes+1, num_of_nodes+2] = Dp
    DD[num_of_nodes+2, 2:num_of_nodes+1] = Dp
    TT[2:num_of_nodes+1, 2:num_of_nodes+1] = T
    TT[2:num_of_nodes+1, 1] = Tp
    TT[1, 2:num_of_nodes+1] = Tp
    TT[2:num_of_nodes+1, num_of_nodes+2] = Tp
    TT[num_of_nodes+2, 2:num_of_nodes+1] = Tp

    return TT, DD
end

function read_data_Agatz(sample::String; HM::Bool=false)
    distribution = split(sample, "-")[1]
    filename = joinpath(@__DIR__, "Test_Instances/TSPD-Instances-Agatz/$(distribution)/$(sample).txt")
    # @show filename
    f = open(filename, "r")
    lines = readlines(f)

    alpha = parse(Float64, lines[2]) / parse(Float64, lines[4])
    n_nodes = parse(Int64, lines[6]) - 1
    depot = (parse(Float64, split(lines[8], " ")[1]), parse(Float64, split(lines[8], " ")[2]))
    customers = Vector{Tuple{Float64,Float64}}()
    for i = 1:n_nodes
        node = (parse(Float64, split(lines[9+i], " ")[1]), parse(Float64, split(lines[9+i], " ")[2]))
        push!(customers, node)
    end

    # Calculate original distance matrices
    T, D = Calculate_distance_matrices_with_dummy_Agatz(alpha, depot, customers)

    if HM
        x_coordinates = [depot[1]; [customer[1] for customer in customers]]
        y_coordinates = [depot[2]; [customer[2] for customer in customers]]
        return x_coordinates, y_coordinates, T, D
    else
        return T, D
    end
end

function read_data_Agatz_restricted(sample::String)
    filename = joinpath(@__DIR__, "Test_Instances/TSPD-Instances-Agatz/restricted/maxradius/$(sample).txt")
    f = open(filename, "r")
    lines = readlines(f)
    flying_range = parse(Float64, split(lines[1], " ")[2])
    alpha = parse(Float64, lines[4]) / parse(Float64, lines[6])
    n_nodes = parse(Int64, lines[8]) - 1
    depot = (parse(Float64, split(lines[10], " ")[1]), parse(Float64, split(lines[10], " ")[2]))
    customers = Vector{Tuple{Float64,Float64}}()
    for i = 1:n_nodes
        node = (parse(Float64, split(lines[11+i], " ")[1]), parse(Float64, split(lines[11+i], " ")[2]))
        push!(customers, node)
    end
    T, D = Calculate_distance_matrices_with_dummy_Agatz(alpha, depot, customers)
    return T, D, flying_range
end

function read_data_Bogyrbayeva(file_name::String, sample_number::Int)
    distribution, _, n_nodes_ = split(file_name, "-")
    n_nodes = parse(Int, n_nodes_)
    filename = joinpath(@__DIR__, "Test_Instances/TSPD-Instances-Bogyrbayeva/$(distribution)/$(file_name).txt")
    f = open(filename, "r")
    lines = readlines(f)
    data = parse.(Float64, split(lines[sample_number], " "))
    depot = [data[1], data[2]]
    customers = zeros(n_nodes - 1, 2)
    for i = 2:n_nodes
        customers[i-1, 1] = data[2*i-1]
        customers[i-1, 2] = data[2*i]
    end
    return depot, customers
end