import zarr
import numpy as np
from typing import Any, Dict, List, Tuple, Union


class MultiDimTensorZarr:
    """
    A class to store and load an N-dimensional tensor in Zarr, along with
    dimension names and IDs for each axis.
    """

    def __init__(
        self,
        data: np.ndarray,
        dims_info: List[Dict[str, Any]],
    ):
        """
        Initialize the MultiDimTensorZarr object with data and dimension info.

        Parameters
        ----------
        data : np.ndarray
            An N-dimensional NumPy array representing your tensor.
        dims_info : List[Dict[str, Any]]
            A list of dictionaries, each describing one dimension/axis.

            Example of dims_info for a 3D array:
            [
                {
                    "name": "time",
                    "ids": ["2020-01", "2020-02", ...]
                },
                {
                    "name": "origin",
                    "ids": ["O1", "O2", ...]
                },
                {
                    "name": "destination",
                    "ids": ["D1", "D2", ...]
                }
            ]

            Each dictionary must have:
              - "name": A string that gives a meaningful label for the dimension.
              - "ids": A list or 1D array that has the same length as data.shape for that axis.
        """
        self._validate_init_args(data, dims_info)

        self.data = data
        self.dims_info = dims_info
        # We'll store dimension count for convenience
        self.n_dims = data.ndim

    def _validate_init_args(self, data: np.ndarray, dims_info: List[Dict[str, Any]]):
        """
        Validate that data and dims_info have matching dimensions.
        """
        if len(dims_info) != data.ndim:
            raise ValueError(
                f"Number of dimensions in dims_info ({len(dims_info)}) "
                f"does not match data.ndim ({data.ndim})."
            )
        for i, dim_dict in enumerate(dims_info):
            if "name" not in dim_dict:
                raise KeyError(f"Missing 'name' key in dims_info[{i}].")
            if "ids" not in dim_dict:
                raise KeyError(f"Missing 'ids' key in dims_info[{i}].")
            dim_ids = np.asarray(dim_dict["ids"])
            if dim_ids.shape[0] != data.shape[i]:
                raise ValueError(
                    f"Dimension {i} (named '{dim_dict['name']}') has {dim_ids.shape[0]} IDs, "
                    f"but data.shape[{i}] is {data.shape[i]}."
                )

    def store_to_zarr(
        self,
        store_path: str,
        group_name: str = "tensor_data",
        compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2),
        chunk_size: Union[None, Tuple[int, ...]] = None,
        overwrite: bool = True,
    ):
        """
        Store the N-dimensional data and dimension info into a Zarr store.

        Parameters
        ----------
        store_path : str
            Path (or URL) to the Zarr store location (e.g., a local directory).
        group_name : str
            Name of the top-level group in the Zarr store.
        compressor : zarr.Codec
            Compressor to use for Zarr array (default is Blosc ZSTD).
        chunk_size : tuple or None
            Chunk shape for the stored array. If None, let Zarr auto-chunk or
            choose a default. Otherwise, must match data.ndim in length.
        overwrite : bool
            Whether to overwrite the existing store at 'store_path'.
        """
        # Create or open the Zarr store
        store = zarr.DirectoryStore(store_path)
        mode = "w" if overwrite else "a"
        root_group = zarr.group(store=store, overwrite=overwrite)

        # Create a subgroup for the data
        tensor_group = root_group.require_group(group_name)

        # Create the array for the data
        tensor_array = tensor_group.empty(
            name="data",
            shape=self.data.shape,
            dtype=self.data.dtype,
            chunks=chunk_size,
            compressor=compressor,
        )

        # Write the data
        tensor_array[:] = self.data

        # Store dimension information
        # Option 1: store each dimension IDs in a separate array
        for i, dim_dict in enumerate(self.dims_info):
            dim_name = dim_dict["name"]
            dim_ids = np.asarray(dim_dict["ids"])

            # We'll store them in a subgroup named after the dimension name
            # or something consistent like f"dim_{i}"
            dim_subgroup = tensor_group.require_group(f"dim_{i}")
            # Example: store dimension info as arrays
            dim_ids_zarr = dim_subgroup.empty(
                name="ids",
                shape=dim_ids.shape,
                dtype=dim_ids.dtype,
            )
            dim_ids_zarr[:] = dim_ids

            # Also store attributes: dimension name, etc.
            dim_subgroup.attrs["name"] = dim_name

        # Option 2 (additionally or alternatively): store dimension metadata
        # as a single attribute in JSON-serializable form
        # (Converting arrays to lists for JSON compatibility)
        dims_info_serializable = []
        for dim_dict in self.dims_info:
            dims_info_serializable.append(
                {
                    "name": dim_dict["name"],
                    "ids": np.asarray(dim_dict["ids"]).tolist(),
                }
            )
        tensor_group.attrs["dims_info"] = dims_info_serializable

    @classmethod
    def load_from_zarr(cls, store_path: str, group_name: str = "tensor_data"):
        """
        Load the N-dimensional data and dimension info from a Zarr store,
        and return a MultiDimTensorZarr instance.

        Parameters
        ----------
        store_path : str
            Path (or URL) to the Zarr store location.
        group_name : str
            Name of the top-level group in the Zarr store.

        Returns
        -------
        MultiDimTensorZarr
            A new instance with the loaded data and dimension metadata.
        """
        # Open the store
        store = zarr.DirectoryStore(store_path)
        root_group = zarr.open_group(store=store, mode="r")

        # Access the subgroup
        tensor_group = root_group[group_name]

        # Load data
        data = tensor_group["data"][:]

        # Attempt to load dimension metadata from attributes or from subgroups
        dims_info_attr = tensor_group.attrs.get("dims_info", None)

        # Reconstruct dims_info
        if dims_info_attr is not None:
            # If we stored them as JSON-serializable data in attributes
            dims_info = []
            for i, dim_dict in enumerate(dims_info_attr):
                # We can cross-check subgroups if we like, or just trust
                # the attribute. For example:
                name = dim_dict["name"]
                ids_list = dim_dict["ids"]
                dims_info.append({"name": name, "ids": ids_list})
        else:
            # Alternatively, we rely on the subgroups named "dim_0", "dim_1", ...
            dims_info = []
            for i in range(data.ndim):
                dim_subgroup = tensor_group[f"dim_{i}"]
                name = dim_subgroup.attrs["name"]
                ids = dim_subgroup["ids"][:]
                dims_info.append({"name": name, "ids": ids})

        # Create and return instance
        return cls(data, dims_info)

    def get_dim_info(self, axis: int) -> Dict[str, Any]:
        """
        Get the dimension info dictionary for a given axis index.

        Parameters
        ----------
        axis : int
            Axis index (0-based).

        Returns
        -------
        dict
            Dictionary containing 'name' and 'ids' for the specified axis.
        """
        if axis < 0 or axis >= self.n_dims:
            raise IndexError(f"Axis {axis} is out of range for an array of {self.n_dims} dims.")
        return self.dims_info[axis]

    def get_value(self, *indices: int) -> Any:
        """
        Get the value at a particular index in the data, while also
        returning the corresponding dimension IDs.

        Example:
        If data is 3D, get_value(2, 5, 1) returns the numeric value
        and the corresponding dimension IDs for each axis.

        Parameters
        ----------
        indices : int
            Indices for each axis.

        Returns
        -------
        (value, [dim_id_for_axis_0, dim_id_for_axis_1, ...])
        """
        if len(indices) != self.n_dims:
            raise ValueError(
                f"Number of indices ({len(indices)}) does not match the "
                f"number of dimensions ({self.n_dims})."
            )

        # Retrieve the value from the data array
        value = self.data[indices]

        # Retrieve the dimension IDs
        dim_ids_for_indices = []
        for axis, idx in enumerate(indices):
            dim_ids = np.asarray(self.dims_info[axis]["ids"])
            dim_ids_for_indices.append(dim_ids[idx])

        return value, dim_ids_for_indices


if __name__ == "__main__":
    ############################################
    # EXAMPLE USAGE
    ############################################

    # Create synthetic 3D data
    np.random.seed(0)
    shape = (2, 3, 4)  # time x origin x destination, for instance
    data_3d = np.random.randint(0, 100, size=shape)

    # Provide dimension info
    dims_info = [
        {
            "name": "time",
            "ids": ["t0", "t1"],  # length 2
        },
        {
            "name": "origin",
            "ids": ["O1", "O2", "O3"],  # length 3
        },
        {
            "name": "destination",
            "ids": ["D1", "D2", "D3", "D4"],  # length 4
        },
    ]

    # Create the MultiDimTensorZarr object
    tensor_obj = MultiDimTensorZarr(data=data_3d, dims_info=dims_info)

    # Store to Zarr
    tensor_obj.store_to_zarr(store_path=r"data/example_3d.zarr")

    # Load from Zarr
    loaded_obj = MultiDimTensorZarr.load_from_zarr(store_path=r"data/example_3d.zarr")

    # Demonstrate usage
    print("Original data shape:", data_3d.shape)
    print("Loaded data shape:", loaded_obj.data.shape)
    print("Dimension 0 info:", loaded_obj.get_dim_info(0))
    print("Dimension 1 info:", loaded_obj.get_dim_info(1))
    print("Dimension 2 info:", loaded_obj.get_dim_info(2))

    # Get a value
    value, ids = loaded_obj.get_value(1, 2, 3)
    print("Value at index (1,2,3):", value)
    print("Corresponding dimension IDs:", ids)
