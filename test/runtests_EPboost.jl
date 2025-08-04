using JSON3

include("read_files.jl")
include("../src/tsp.jl")
include("../src/TSPDChain.jl")
include("../src/utils.jl")
include("../src/tsp_ep_all.jl")



function depot_customer_to_tour(depot, customers)
    n = size(customers)[1] + 1
    x, y = zeros(n), zeros(n)
    x[1], y[1] = depot
    for i in 1:n-1
        x[i+1], y[i+1] = customers[i, :]
    end
    return x, y
end


function warmup_Bogyrbayeva(s)
    depot, customers = read_data_Bogyrbayeva(s, 1)
    x, y = depot_customer_to_tour(depot, customers)
    Ct, Cd = cost_matrices_with_dummy(x, y, 1.0, 1.0)
    init_tour = collect(1:size(x, 1)+1)
    tsp_ep_all_original(Ct, Cd, init_tour)
    tsp_ep_all_boosted(Ct, Cd, init_tour)
end

function generate_concorde_tour_Bogyrbayeva()
    concorde_tours = Dict()
    for s in ["Amsterdam-n_nodes-10", "Amsterdam-n_nodes-20", "Amsterdam-n_nodes-50", "Random-n_nodes-20", "Random-n_nodes-50", "Random-n_nodes-100"]
        concorde_tours[s] = Dict{Int, Vector{Int}}()
        for i in 1:100
            println(i)
            depot, customers = read_data_Bogyrbayeva(s, 1)
            x, y = depot_customer_to_tour(depot, customers)
            Ct, _ = cost_matrices_with_dummy(x, y, 1.0, 1.0)
            init_tour = concorde_with_dummy_nodes(Ct)
            concorde_tours[s][i] = init_tour
        end
    end
    open("test/concorde_tours_Bogyrbayeva.json", "w") do file
        JSON3.write(file, concorde_tours)
    end
    return concorde_tours
end

function test_Bogyrbayeva_EP_boost(concorde_tours)
    data = Dict("Amsterdam" => Dict(), "Random" => Dict())
    for s in ["Amsterdam-n_nodes-10", "Amsterdam-n_nodes-20", "Amsterdam-n_nodes-50", "Random-n_nodes-20", "Random-n_nodes-50", "Random-n_nodes-100"]
        distribution, _, n_nodes = split(s, "-")
        n_nodes = parse(Int, n_nodes)
        if !haskey(data, distribution)
            data[distribution] = Dict()
        end
        data[distribution][n_nodes] = Dict(1 => Dict(), 2 => Dict(), 3 => Dict(), "average_time" => 0.0, "average_time_boosted" => 0.0)
        warmup_Bogyrbayeva(s)

        for alpha in [1, 2, 3]
            total, total_boosted = 0.0, 0.0
            for i = 1:100
                println("Instance: $s, Alpha: $alpha, Iteration: $i")
                depot, customers = read_data_Bogyrbayeva(s, 1)
                x, y = depot_customer_to_tour(depot, customers)
                Ct, Cd = cost_matrices_with_dummy(x, y, 1.0, 1.0/alpha)
                init_tour = concorde_tours[s][i]

                (cost, troute, droute), elapsed, _, _, _ = @timed tsp_ep_all_original(Ct, Cd, init_tour)
                (cost_boosted, troute_boosted, droute_boosted), elapsed_boosted, _, _, _ = @timed tsp_ep_all_boosted(Ct, Cd, init_tour)
                @assert isapprox(cost, cost_boosted)
                
                total += elapsed
                total_boosted += elapsed_boosted

                data[distribution][n_nodes][alpha][i] = Dict(
                    :init_tour => init_tour,
                    :cost => cost,
                    :elapsed => elapsed,
                    :troute => troute,
                    :droute => droute,
                    :cost_boosted => cost_boosted,
                    :elapsed_boosted => elapsed_boosted,
                    :troute_boosted => troute_boosted,
                    :droute_boosted => droute_boosted
                )
            end
            data[distribution][n_nodes][alpha]["average_time"]  = total / 100
            data[distribution][n_nodes][alpha]["average_time_boosted"] = total_boosted / 100
            println("Instance: $s, Alpha: $alpha, Average time: $(total / 100), Average time (boosted): $(total_boosted / 100)")
        end
        data[distribution][n_nodes]["average_time"] = sum(data[distribution][n_nodes][alpha]["average_time"] for alpha in [1, 2, 3]) / 3
        data[distribution][n_nodes]["average_time_boosted"] = sum(data[distribution][n_nodes][alpha]["average_time_boosted"] for alpha in [1, 2, 3]) / 3
        println("Instance: $s, Average time: $(data[distribution][n_nodes]["average_time"]), Average time (boosted): $(data[distribution][n_nodes]["average_time_boosted"])")

        open("test/Bogyrbayeva_EP_boosted.json", "w") do file
            JSON3.write(file, data)
        end
    end
end



function warmup_Agatz(distribution, n)
    for alpha in [1, 2, 3]
        files = readdir("test/Test_Instances/TSPD-Instances-Agatz/$distribution", join=true)  # Adjust this path to match your directory structure
        pattern = alpha == 2 ? "$distribution-\\d+-n$n\\b" : "$distribution-alpha_$alpha-\\d+-n$n\\b"
        regex = Regex(pattern)

        for filename in files
            m = match(regex, filename)
            if !isnothing(m)
                Ct, Cd = read_data_Agatz(String(m.match))
                init_tour = collect(1:n+1)
                tsp_ep_all_original(Ct, Cd, init_tour)
                tsp_ep_all_boosted(Ct, Cd, init_tour)
                return
            end
        end
    end
end

function generate_concorde_tour_Agatz()
    concorde_tours = Dict()
    for distribution in ["doublecenter", "singlecenter", "uniform"]
        for n in [10, 20, 50, 75, 100, 175, 250, 375, 500]
            for alpha in [1, 2, 3]
                files = readdir("test/Test_Instances/TSPD-Instances-Agatz/$distribution", join=true)  # Adjust this path to match your directory structure
                pattern = alpha == 2 ? "$distribution-\\d+-n$n\\b" : "$distribution-alpha_$alpha-\\d+-n$n\\b"
                regex = Regex(pattern)
                for filename in files
                    m = match(regex, filename)
                    if !isnothing(m)
                        Ct, _ = read_data_Agatz(String(m.match))
                        concorde_tours[filename] = concorde_with_dummy_nodes(Ct)
                    end
                end
            end
        end
    end
    open("test/concorde_tours_Agatz.json", "w") do file
        JSON3.write(file, concorde_tours)
    end
    return concorde_tours
end

function test_Agatz_EP_boost(concorde_tours)
    data = Dict("doublecenter" => Dict(), "singlecenter" => Dict(), "uniform" => Dict())
    for n in [10, 20, 50, 75, 100, 175, 250, 375, 500]
        for distribution in ["doublecenter", "singlecenter", "uniform"]
            data[distribution][n] = Dict()
            warmup_Agatz(distribution, n)
            total_over_alpha, total_boosted_over_alpha, total_cnt_over_alpha = 0.0, 0.0, 0

            for alpha in [1, 2, 3]
                println("Distribution: $distribution, n: $n, Alpha: $alpha")

                data[distribution][n][alpha] = Dict()
                total, total_boosted, total_cnt = 0.0, 0.0, 0

                files = readdir("test/Test_Instances/TSPD-Instances-Agatz/$distribution", join=true)  # Adjust this path to match your directory structure
                pattern = alpha == 2 ? "$distribution-\\d+-n$n\\b" : "$distribution-alpha_$alpha-\\d+-n$n\\b"
                regex = Regex(pattern)
                for filename in files
                    m = match(regex, filename)
                    if !isnothing(m)
                        println("Processing file: $filename")

                        Ct, Cd = read_data_Agatz(String(m.match))
                        init_tour = concorde_tours[filename]

                        (cost, troute, droute), elapsed, _, _, _ = @timed tsp_ep_all_original(Ct, Cd, init_tour)
                        (cost_boosted, troute_boosted, droute_boosted), elapsed_boosted, _, _, _ = @timed tsp_ep_all_boosted(Ct, Cd, init_tour)
                        # @assert isapprox(cost, cost_boosted)
                        
                        total += elapsed
                        total_boosted += elapsed_boosted
                        total_cnt += 1

                        _, i, _ = split(filename, "-")
                        data[distribution][n][alpha][i] = Dict(
                            :init_tour => init_tour,
                            :cost => cost,
                            :elapsed => elapsed,
                            :troute => troute,
                            :droute => droute,
                            :cost_boosted => cost_boosted,
                            :elapsed_boosted => elapsed_boosted,
                            :troute_boosted => troute_boosted,
                            :droute_boosted => droute_boosted
                        )
                    end
                end
                if total_cnt != 0
                    data[distribution][n][alpha]["average_time"]  = total / total_cnt
                    data[distribution][n][alpha]["average_time_boosted"] = total_boosted / total_cnt
                    data[distribution][n][alpha]["total_cnt"] = total_cnt
                    total_over_alpha += total
                    total_boosted_over_alpha += total_boosted
                    total_cnt_over_alpha += total_cnt
                    println("Distribution: $distribution, n: $n, Alpha: $alpha, Average time: $(total / total_cnt), Average time (boosted): $(total_boosted / total_cnt)")

                    open("test/Agatz_EP_boosted.json", "w") do file
                        JSON3.write(file, data)
                    end
                end
            end
            if total_cnt_over_alpha != 0
                data[distribution][n]["average_time"] = total_over_alpha / total_cnt_over_alpha
                data[distribution][n]["average_time_boosted"] = total_boosted_over_alpha / total_cnt_over_alpha
                data[distribution][n]["total_cnt"] = total_cnt_over_alpha
                println("Distribution: $distribution, n: $n, Average time: $(total_over_alpha / total_cnt_over_alpha), Average time (boosted): $(total_boosted_over_alpha / total_cnt_over_alpha)")
            end
        end
    end
end

# concorde_tours_bogyrbayeva = generate_concorde_tour_Bogyrbayeva()
# concorde_tours_bogyrbayeva = JSON3.read("test/concorde_tours_Bogyrbayeva.json", Dict{String, Dict{Int, Vector{Int}}})
# test_Bogyrbayeva_EP_boost(concorde_tours_bogyrbayeva)

# concorde_tours_agatz = generate_concorde_tour_Agatz()
concorde_tours_agatz = JSON3.read("test/concorde_tours_Agatz.json", Dict{String, Vector{Int}})
test_Agatz_EP_boost(concorde_tours_agatz)