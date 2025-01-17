import numpy as np
from TNT_Tensor import MultiDimTensorZarr

def main():
    # Suppose we have 5 origins (O1..O5) and 4 destinations (D1..D4).
    n_origins = 5
    n_destinations = 4

    # Create a 2D numpy array of travel times (e.g., in minutes).
    # Shape = (n_origins, n_destinations)
    travel_times = np.random.randint(low=5, high=60, size=(n_origins, n_destinations))

    # Build dimension info:
    #   dims_info[0] -> origin dimension
    #   dims_info[1] -> destination dimension
    dims_info = [
        {
            "name": "O",  # 'Origin' dimension name
            "ids": [f"O{i+1}" for i in range(n_origins)]
        },
        {
            "name": "D",  # 'Destination' dimension name
            "ids": [f"D{j+1}" for j in range(n_destinations)]
        },
    ]

    # Instantiate MultiDimTensorZarr
    tensor = MultiDimTensorZarr(data=travel_times, dims_info=dims_info)

    # Store to a Zarr directory
    zarr_path = "travel_time.zarr"
    tensor.store_to_zarr(store_path=zarr_path, group_name="travel_times_2d")

    # Load it back from the Zarr store
    loaded_tensor = MultiDimTensorZarr.load_from_zarr(
        store_path=zarr_path,
        group_name="travel_times_2d"
    )

    # Print shapes and dimension info
    print("Original shape:", travel_times.shape)
    print("Loaded shape:  ", loaded_tensor.data.shape)

    print("Dimension 0 info (Origins):", loaded_tensor.get_dim_info(0))
    print("Dimension 1 info (Destinations):", loaded_tensor.get_dim_info(1))

    # Demonstrate retrieving a single value with its dimension labels
    # For example, value at (origin=2, destination=1)
    index_origin = 2
    index_destination = 1
    val, labels = loaded_tensor.get_value(index_origin, index_destination)
    print(f"Travel time at indices (2,1): {val} minutes")
    print("Corresponding dimension labels:", labels)


if __name__ == "__main__":
    main()
