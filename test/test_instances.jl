
function Show_Benchmark_Count_Agatz_unrestricted(distribution::String, alpha::Int, n::Int)
    total_objective_value = 0.0
    total_time = 0.0
    count = 0
    files = readdir("Test_Instances/TSPD-Instances-Agatz/$distribution", join=true)  # Adjust this path to match your directory structure
    pattern = alpha == 2 ? "$distribution-\\d+-n$n\\b" : "$distribution-alpha_$alpha-\\d+-n$n\\b"  # Construct the filename pattern from the inputs
    # @show pattern
    regex = Regex(pattern)
    # Print all filenames and their match status
    # println("Files in directory and their match status:")
    for filename in files
        m = match(regex, filename)  # Use the match function
        println("Filename: $filename, Match: $m")
        if m !== nothing  # Check if a match was found
            count += 1
        end
    end
    @show count
end

# ==================================================================================================
# Test Algorithms (ICP, NICP)
# ==================================================================================================

function Solve_Agatz_unrestricted(sample::String;     
    problem_type::Symbol=:TSPD,
    flying_range::Float64=Inf, 
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy)
    
    T, D = read_data_Agatz(sample)
    
    return solve_tspd_by_ICP(T, D;
    problem_type=problem_type,     
    flying_range=flying_range, 
    max_nodes=max_nodes, 
    chain_initialization_method=chain_initialization_method,
    chainlet_initialization_method=chainlet_initialization_method, 
    chainlet_solving_method=chainlet_solving_method,
    chainlet_evaluation_method=chainlet_evaluation_method,
    search_method=search_method)
end

function Solve_Agatz_restricted(sample::String;     
    problem_type::Symbol=:TSPD,
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy)
    
    T, D, flying_range = read_data_Agatz_restricted(sample)

    return solve_tspd_by_ICP(T, D;
    problem_type=problem_type, 
    flying_range=flying_range,
    max_nodes=max_nodes, 
    chain_initialization_method=chain_initialization_method,
    chainlet_initialization_method=chainlet_initialization_method, 
    chainlet_solving_method=chainlet_solving_method,
    chainlet_evaluation_method=chainlet_evaluation_method,
    search_method=search_method)
end

function Solve_Bogyrbayeva(file_name::String, sample_number::Int64;     
    problem_type::Symbol=:TSPD,
    flying_range::Float64=Inf, 
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy)
    
    depot, customers = read_data_Bogyrbayeva(file_name, sample_number)
    
    return solve_tspd_by_ICP(depot, customers, 1.0, 2.0; 
    problem_type=problem_type,
    flying_range=flying_range, 
    max_nodes=max_nodes, 
    chain_initialization_method=chain_initialization_method,
    chainlet_initialization_method=chainlet_initialization_method, 
    chainlet_solving_method=chainlet_solving_method,
    chainlet_evaluation_method=:Default,
    search_method=search_method)
end

function test_Agatz_unrestricted(distribution::String, alpha::Int, n::Int; 
    problem_type::Symbol=:TSPD,
    flying_range::Float64=Inf, 
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy,
    verbose::Bool=false)

    if chainlet_evaluation_method == :Neuro
        print("NICP : ")
    else
        print("ICP : ")
    end

    total_objective_value = 0.0
    total_best_objective_value = 0.0
    total_time = 0.0
    count = 0
    files = readdir("Test_Instances/TSPD-Instances-Agatz/$distribution", join=true)  # Adjust this path to match your directory structure
    pattern = alpha == 2 ? "$distribution-\\d+-n$n\\b" : "$distribution-alpha_$alpha-\\d+-n$n\\b"  # Construct the filename pattern from the inputs
    # @show pattern
    regex = Regex(pattern)
    # Print all filenames and their match status
    # println("Files in directory and their match status:")
    for filename in files
        m = match(regex, filename)  # Use the match function
        # println("Filename: $filename, Match: $m")
        if m !== nothing  # Check if a match was found
            # @show m

            time = @elapsed result_chain = Solve_Agatz_unrestricted(String(m.match);  
            problem_type=problem_type,
            flying_range=flying_range, 
            max_nodes=max_nodes, 
            chain_initialization_method=chain_initialization_method,
            chainlet_initialization_method=chainlet_initialization_method, 
            chainlet_solving_method=chainlet_solving_method,
            chainlet_evaluation_method=chainlet_evaluation_method,
            search_method=search_method)

            # Validation
            validate_tspd_chain(result_chain)

            obj_value = result_chain.objective_value
            total_objective_value += obj_value 
            total_time += time
            count += 1
        
            if verbose
                println("Instance: $count, Objective Value: $obj_value, Time: $time")
                # @show result_chain.truck_route, result_chain.drone_route
            end
        end
    end
    avg_time = total_time / count
    avg_objective_value = total_objective_value / count
    println("Average time: $avg_time, Average objective value: $avg_objective_value")
end

function test_Agatz_restricted(n::Int, maxradius::Int; 
    problem_type::Symbol=:TSPD,
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy,
    verbose::Bool=false)
    
    total_objective_value = 0.0
    total_time = 0.0
    count = 0
    files = readdir("Test_Instances/TSPD-Instances-Agatz/restricted/maxradius", join=true)  # Adjust this path to match your directory structure
    pattern = "uniform-\\d+-n$n-maxradius-$maxradius"  # Construct the filename pattern from the inputs
    regex = Regex(pattern)
    for filename in files
        m = match(regex, filename)  # Use the match function
        if m !== nothing  # Check if a match was found
            @show m
            time = @elapsed result_chain = Solve_Agatz_restricted(String(m.match); 
            problem_type=problem_type,
            max_nodes=max_nodes, 
            chain_initialization_method=chain_initialization_method,
            chainlet_initialization_method=chainlet_initialization_method, 
            chainlet_solving_method=chainlet_solving_method,
            chainlet_evaluation_method=chainlet_evaluation_method,
            search_method=search_method)
                
            obj_value = result_chain.objective_value
            total_objective_value += obj_value 
            total_time += time
            count += 1

            if verbose
                println("Instance: $count, Objective Value: $obj_value, Time: $time")
                # @show result_chain.truck_route, result_chain.drone_route
            end

            # Validation
            validate_tspd_chain(result_chain)
        end
    end
    avg_time = total_time / count
    avg_objective_value = total_objective_value / count
    println("Average time: $avg_time, Average objective value: $avg_objective_value")
end

function test_Bogyrbayeva(base_name::String, num_nodes::Int64; 
    problem_type::Symbol=:TSPD,
    flying_range::Float64=Inf, 
    max_nodes::Int=20, 
    chain_initialization_method::Symbol=:Concorde,
    chainlet_initialization_method::Symbol=:FI, 
    chainlet_solving_method::Symbol=:TSP_EP_all,
    chainlet_evaluation_method::Symbol=:Default,
    search_method::Symbol=:Greedy,
    verbose::Bool=false)

    if chainlet_evaluation_method == :Neuro
        print("NICP : ")
    else
        print("ICP : ")
    end
    
    total_objective_value = 0.0
    total_time = 0.0
    count = 0
    file_name = "$base_name-n_nodes-$num_nodes"
    for i = 1:100
        time = @elapsed result_chain = Solve_Bogyrbayeva(file_name, i; 
        problem_type=problem_type,
        flying_range=flying_range, 
        max_nodes=max_nodes, 
        chain_initialization_method=chain_initialization_method,
        chainlet_initialization_method=chainlet_initialization_method, 
        chainlet_solving_method=chainlet_solving_method,
        chainlet_evaluation_method=chainlet_evaluation_method,
        search_method=search_method)
        
        obj_value = result_chain.objective_value
        total_objective_value += obj_value 
        total_time += time
        count += 1
        if verbose
            println("Instance: $count, Objective Value: $obj_value, Time: $time")
            # @show result_chain.truck_route, result_chain.drone_route
        end

        # Validation
        validate_tspd_chain(result_chain)
    end
    avg_time = total_time / count
    avg_objective_value = total_objective_value / count
    println("Average time: $avg_time, Average objective value: $avg_objective_value")
end