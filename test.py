import torch

file_path = "stats.pt"
data = torch.load(file_path, weights_only=False, map_location=torch.device('cpu'))
## To check UQ metrics content of stats.pt binary file 
print(data)