import numpy as np
import re
import os
import torch

def load_single_graph(graph_file):
    num_nodes=0
    types = []
    coordinates = []
    edges = []
    edge_weights = []
    weight_pattern = re.compile(r"'weight':\s*([\d.]+)")  # Regex to extract the weight value

    typemapper={"6":0,"1":1,"8":2,"7":3}

    with open(graph_file, 'r') as f:
        reading_edges=True
        for line in f:
            if line=="---\n" :
                reading_edges=False
            else :
                try :
                    if reading_edges :
                        parts = line.strip().split()
                        u, v = int(parts[0]), int(parts[1])

                        # Match the weight using regex
                        match = weight_pattern.search(line)
                        if match:
                            weight = float(match.group(1))  # Extract and convert weight to float
                        else:
                            raise ValueError(f"Could not parse weight from line: {line}")

                        edges.append((u, v, weight))
                    else :
                        parts = line.split()
                        num_nodes+=1
                        types.append(typemapper[parts[1]])
                        coordinates.append([float(parts[2]),float(parts[3]),float(parts[4])])
                except Exception as e:
                        print(f"Error parsing line: {line}")
                        raise e
                    
    num_nodes=max([max(a,b) for a,b,_ in edges])+1

    coordinates = torch.tensor(coordinates,dtype=torch.float)
    types = torch.tensor(types)
    
    return {"e":edges,"r":edge_weights,"t":types,"s":coordinates}

def load_data_from_directory():
    graphs=[]

    for subfolder in os.listdir("graphs") :
        for file in os.listdir("graphs/"+subfolder) :
            graphs.append(load_single_graph("graphs/"+subfolder+"/"+file))

    return graphs

    