
# Internal functions to run ICP and NICP
function solve_tspd_by_ICP(
    T::Matrix{Float64}, 
    D::Matrix{Float64};
    problem_type::Symbol=:TSPD,
    flying_range::Float64=Inf, 
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy
)   
    
    return run_ICP(T, D; 
    problem_type=problem_type,
    flying_range=flying_range, 
    max_nodes=max_nodes, 
    chain_initialization_method=chain_initialization_method,
    chainlet_initialization_method=chainlet_initialization_method, 
    chainlet_solving_method=chainlet_solving_method,
    chainlet_evaluation_method=chainlet_evaluation_method,
    search_method=search_method)
end

function solve_tspd_by_ICP(
    depot::Vector{Float64}, 
    Customers::Matrix{Float64},
    tspeed::Float64,
    dspeed::Float64;
    problem_type::Symbol=:TSPD,
    flying_range::Float64=Inf, 
    max_nodes::Int=20,
    chain_initialization_method::Symbol=:Concorde, 
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy
)

    T, D = calculate_duration_matrices(tspeed, dspeed, depot, Customers)

    return run_ICP(T, D; 
    problem_type=problem_type,
    flying_range=flying_range, 
    max_nodes=max_nodes, 
    chain_initialization_method=chain_initialization_method,
    chainlet_initialization_method=chainlet_initialization_method, 
    chainlet_solving_method=chainlet_solving_method,
    chainlet_evaluation_method=chainlet_evaluation_method,
    search_method=search_method)
end

