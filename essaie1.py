from utils import generate_plots

list_of_dirs = [
    "logs/mlp_relu", 
    "logs/mlp_sigmoid",
    "logs/mlp_tanh",
    "logs/mlp_leaky_relu"
]

legend_names = ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU"]
save_path = "plots"

generate_plots(list_of_dirs, legend_names, save_path)
