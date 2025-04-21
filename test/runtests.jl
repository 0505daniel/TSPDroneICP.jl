include("../src/main.jl")

# using TSPDroneICP

include("read_files.jl")
include("test_instances.jl")

# Solve_Agatz_unrestricted("uniform-alpha_1-75-n50")   
# Solve_Agatz_unrestricted("uniform-alpha_1-75-n50"; chainlet_evaluation_method=:Neuro)   
# Solve_Agatz_restricted("uniform-62-n20-maxradius-20")  
Solve_Bogyrbayeva("Amsterdam-n_nodes-50", 1)  # instance number 1
# Solve_Bogyrbayeva("Amsterdam-n_nodes-50", 1; chainlet_evaluation_method=:Neuro)  # instance number 1