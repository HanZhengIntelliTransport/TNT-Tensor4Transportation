import numpy as np
import pandas as pd
import networkx as nx

# Generate a sample network with 10 nodes and random edges
G = nx.random_geometric_graph(10, 0.5)
pos = nx.get_node_attributes(G, 'pos')  # Extract node positions

# Create node attributes
for node in G.nodes:
    G.nodes[node]['name'] = f"Location_{node}"

# Define tensor dimensions
locations = [f"Location_{i}" for i in range(len(G.nodes))]
weather_conditions = ['Sunny', 'Rainy', 'Snowy', 'Cloudy']
congestion_levels = ['Low', 'Medium', 'High']
dates = pd.date_range(start="2025-01-01", periods=7).tolist()
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a tensor: OD matrix with other dimensions
tensor_shape = (len(locations), len(locations), len(weather_conditions), len(congestion_levels), len(dates))
travel_data_tensor = np.full(tensor_shape, np.nan)  # Initialize with NaN for missing data

# Fill some data randomly
np.random.seed(42)
for _ in range(100):  # Populate 100 random entries
    o, d = np.random.choice(len(locations), 2, replace=False)
    weather = np.random.choice(len(weather_conditions))
    congestion = np.random.choice(len(congestion_levels))
    date = np.random.choice(len(dates))
    travel_data_tensor[o, d, weather, congestion, date] = np.random.randint(50, 200)  # Random travel time

# Display summary of the generated data
tensor_dimensions = {
    "Origin Locations": len(locations),
    "Destination Locations": len(locations),
    "Weather Conditions": len(weather_conditions),
    "Congestion Levels": len(congestion_levels),
    "Dates": len(dates),
}
summary = pd.DataFrame(list(tensor_dimensions.items()), columns=["Dimension", "Size"])


import zarr

# Save the tensor using Zarr
zarr_file_path = "data/sample/travel_data_tensor.zarr"
zarr_store = zarr.DirectoryStore(zarr_file_path)
root = zarr.group(store=zarr_store)

# Save tensor and metadata
root.create_dataset("travel_data_tensor", data=travel_data_tensor, overwrite=True)
root.create_dataset("locations", data=np.array(locations, dtype=str), dtype="str", overwrite=True)
root.create_dataset("weather_conditions", data=np.array(weather_conditions, dtype=str), dtype="str", overwrite=True)
root.create_dataset("congestion_levels", data=np.array(congestion_levels, dtype=str), dtype="str", overwrite=True)
root.create_dataset("dates", data=np.array([str(date) for date in dates], dtype=str), dtype="str", overwrite=True)
root.create_dataset("days_of_week", data=np.array(days_of_week, dtype=str), dtype="str", overwrite=True)