import zarr
import numpy as np
from typing import Any, Dict, List, Tuple, Union


class MultiDimTensorZarr:
    """
    A class to store and load an N-dimensional tensor in Zarr, along with
    dimension names and IDs for each axis, and store/load regression results.
    """

    def __init__(
        self,
        data: np.ndarray,
        dims_info: List[Dict[str, Any]],
    ):
        self._validate_init_args(data, dims_info)
        self.data = data
        self.dims_info = dims_info
        self.n_dims = data.ndim

    def _validate_init_args(self, data: np.ndarray, dims_info: List[Dict[str, Any]]):
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
        store = zarr.DirectoryStore(store_path)
        root_group = zarr.group(store=store, overwrite=overwrite)
        tensor_group = root_group.require_group(group_name)

        tensor_array = tensor_group.empty(
            name="data",
            shape=self.data.shape,
            dtype=self.data.dtype,
            chunks=chunk_size,
            compressor=compressor,
        )
        tensor_array[:] = self.data

        # Store dimension info in subgroups
        for i, dim_dict in enumerate(self.dims_info):
            dim_name = dim_dict["name"]
            dim_ids = np.asarray(dim_dict["ids"])
            dim_subgroup = tensor_group.require_group(f"dim_{i}")
            dim_ids_zarr = dim_subgroup.empty(
                name="ids",
                shape=dim_ids.shape,
                dtype=dim_ids.dtype,
            )
            dim_ids_zarr[:] = dim_ids
            dim_subgroup.attrs["name"] = dim_name

        # Store dimension info as attribute
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
        store = zarr.DirectoryStore(store_path)
        root_group = zarr.open_group(store=store, mode="r")
        tensor_group = root_group[group_name]

        data = tensor_group["data"][:]
        dims_info_attr = tensor_group.attrs.get("dims_info", None)

        if dims_info_attr is not None:
            dims_info = []
            for i, dim_dict in enumerate(dims_info_attr):
                name = dim_dict["name"]
                ids_list = dim_dict["ids"]
                dims_info.append({"name": name, "ids": ids_list})
        else:
            # fallback
            dims_info = []
            for i in range(data.ndim):
                dim_subgroup = tensor_group[f"dim_{i}"]
                name = dim_subgroup.attrs["name"]
                ids = dim_subgroup["ids"][:]
                dims_info.append({"name": name, "ids": ids})

        return cls(data, dims_info)

    def get_dim_info(self, axis: int) -> Dict[str, Any]:
        if axis < 0 or axis >= self.n_dims:
            raise IndexError(f"Axis {axis} is out of range for an array of {self.n_dims} dims.")
        return self.dims_info[axis]

    def get_value(self, *indices: int) -> Any:
        if len(indices) != self.n_dims:
            raise ValueError(
                f"Number of indices ({len(indices)}) does not match the "
                f"number of dimensions ({self.n_dims})."
            )
        value = self.data[indices]
        dim_ids_for_indices = []
        for axis, idx in enumerate(indices):
            dim_ids = np.asarray(self.dims_info[axis]["ids"])
            dim_ids_for_indices.append(dim_ids[idx])
        return value, dim_ids_for_indices

    # -------------------------------------------------------------------------
    # NEW / UPDATED METHODS: For Tensor Regression Results and OD Slicing
    # -------------------------------------------------------------------------

    def store_regression_result(
        self,
        store_path: str,
        group_name: str = "tensor_data",
        result_group_name: str = "regression_results",
        factor_matrices: List[np.ndarray] = None,
        weights: Union[np.ndarray, None] = None,
        reconstructed_data: Union[np.ndarray, None] = None,
        reconstructed_dims_info: Union[List[Dict[str, Any]], None] = None
    ):
        """
        Store the results of a tensor regression or decomposition (factor matrices,
        weights, or reconstructed data) into a new subgroup under the same store.

        Parameters
        ----------
        store_path : str
            Path to the same (or new) Zarr directory where the original data is stored.
        group_name : str
            The group name where the original data is stored (must exist).
        result_group_name : str
            Name of the new subgroup under `group_name` to store regression results.
        factor_matrices : list of np.ndarray, optional
            A list of factor matrices from a CP or Tucker decomposition, etc.
        weights : np.ndarray, optional
            A weights vector for CP decomposition, if applicable.
        reconstructed_data : np.ndarray, optional
            A fully reconstructed array from the regression. If you want to be able
            to slice this by dimension names, it should have the same shape or a
            sub-shape consistent with the original.
        reconstructed_dims_info : list of dict, optional
            If you want to slice the reconstructed_data by dimension names,
            you should provide dims_info for it. If omitted, we won't store
            dimension info for the reconstructed data.
        """
        store = zarr.DirectoryStore(store_path)
        root_group = zarr.open_group(store=store, mode="a")

        if group_name not in root_group:
            raise ValueError(f"Group '{group_name}' not found in Zarr store.")

        main_group = root_group[group_name]
        results_group = main_group.require_group(result_group_name)

        # 1) Store factor matrices
        if factor_matrices is not None:
            fm_group = results_group.require_group("factor_matrices")
            # Overwrite or store each factor matrix as a dataset
            for i, mat in enumerate(factor_matrices):
                ds_name = f"matrix_{i}"
                if ds_name in fm_group:
                    del fm_group[ds_name]  # remove old data if overwriting
                fm = fm_group.empty(
                    name=ds_name,
                    shape=mat.shape,
                    dtype=mat.dtype,
                )
                fm[:] = mat

        # 2) Store weights
        if weights is not None:
            if "weights" in results_group:
                del results_group["weights"]
            w_arr = results_group.empty(
                name="weights",
                shape=weights.shape,
                dtype=weights.dtype,
            )
            w_arr[:] = weights

        # 3) Store reconstructed data + optional dims_info
        if reconstructed_data is not None:
            rec_group = results_group.require_group("reconstructed_data")
            # Overwrite if it exists
            if "data" in rec_group:
                del rec_group["data"]
            rec_array = rec_group.empty(
                name="data",
                shape=reconstructed_data.shape,
                dtype=reconstructed_data.dtype,
            )
            rec_array[:] = reconstructed_data

            # If dims info is provided, store it so we can reconstruct a MultiDimTensorZarr
            if reconstructed_dims_info is not None:
                # store as attribute
                # convert to JSON-friendly format
                dims_info_serializable = []
                for dim_dict in reconstructed_dims_info:
                    dims_info_serializable.append(
                        {
                            "name": dim_dict["name"],
                            "ids": np.asarray(dim_dict["ids"]).tolist(),
                        }
                    )
                rec_group.attrs["dims_info"] = dims_info_serializable

    @classmethod
    def load_regression_result(
        cls,
        store_path: str,
        group_name: str = "tensor_data",
        result_group_name: str = "regression_results"
    ) -> dict:
        """
        Load any stored factor matrices, weights, and reconstructed data
        from the specified regression_results subgroup.

        Returns a dictionary with keys:
          {
            "factor_matrices": list of np.ndarray or None,
            "weights": np.ndarray or None,
            "reconstructed": MultiDimTensorZarr or None
          }

        If reconstructed data was stored with dimension info,
        we return a MultiDimTensorZarr object. Otherwise, we just return None
        for "reconstructed" or a raw np.ndarray if you prefer.

        Parameters
        ----------
        store_path : str
            Path to Zarr directory.
        group_name : str
            The group where your original (or base) data is stored.
        result_group_name : str
            The name of the subgroup containing regression results.

        Returns
        -------
        dict
            A dictionary with the following possible keys:
            - "factor_matrices": list of loaded factor matrices, if any
            - "weights": loaded weights array, if any
            - "reconstructed": a MultiDimTensorZarr object with reconstructed data
              (if dims_info was stored), else None
        """
        store = zarr.DirectoryStore(store_path)
        root_group = zarr.open_group(store=store, mode="r")

        if group_name not in root_group:
            raise ValueError(f"Group '{group_name}' not found in Zarr store.")

        main_group = root_group[group_name]
        if result_group_name not in main_group:
            raise ValueError(f"No regression results found at '{result_group_name}'.")

        results_group = main_group[result_group_name]

        out_dict = {
            "factor_matrices": None,
            "weights": None,
            "reconstructed": None
        }

        # Load factor_matrices if present
        if "factor_matrices" in results_group:
            fm_group = results_group["factor_matrices"]
            # We don't know how many exist, so we'll collect them all
            factor_mats = []
            for ds_name in sorted(fm_group.keys()):
                mat_data = fm_group[ds_name][:]
                factor_mats.append(mat_data)
            out_dict["factor_matrices"] = factor_mats

        # Load weights if present
        if "weights" in results_group:
            out_dict["weights"] = results_group["weights"][:]

        # Load reconstructed data if present
        if "reconstructed_data" in results_group:
            rec_group = results_group["reconstructed_data"]
            if "data" in rec_group:
                rec_data = rec_group["data"][:]
                # Check if we have dims_info attribute
                dims_info_attr = rec_group.attrs.get("dims_info", None)
                if dims_info_attr is not None:
                    # Reconstruct the dims_info
                    rec_dims_info = []
                    for dim_dict in dims_info_attr:
                        rec_dims_info.append({
                            "name": dim_dict["name"],
                            "ids": dim_dict["ids"]
                        })
                    # Create a MultiDimTensorZarr object
                    rec_tensor = cls(rec_data, rec_dims_info)
                    out_dict["reconstructed"] = rec_tensor
                else:
                    # If no dims_info, user can interpret rec_data as raw np.ndarray
                    # For demonstration, let's just store None or you can store rec_data
                    # if you prefer out_dict["reconstructed_data"] = rec_data
                    pass

        return out_dict

    def extract_subtensor(self, fixed_conditions: Dict[str, Any]) -> "MultiDimTensorZarr":
        """
        Create a sub-tensor by fixing certain dimension(s) to specified ID values.
        Returns a NEW MultiDimTensorZarr object.
        """
        data = self.data
        dims_info = self.dims_info

        index_slices = [slice(None)] * self.n_dims
        new_dims_info = []

        for dim_idx, dim_dict in enumerate(dims_info):
            dim_name = dim_dict["name"]
            dim_ids = np.asarray(dim_dict["ids"])
            if dim_name in fixed_conditions:
                desired_val = fixed_conditions[dim_name]
                matches = np.where(dim_ids == desired_val)[0]
                if len(matches) == 0:
                    raise ValueError(
                        f"No match for value '{desired_val}' in dimension '{dim_name}'."
                    )
                idx_val = matches[0]
                index_slices[dim_idx] = idx_val
            else:
                # keep dimension
                new_dims_info.append(dim_dict)

        sub_data = data[tuple(index_slices)]
        return MultiDimTensorZarr(sub_data, new_dims_info)

    def extract_od_slice(
        self,
        fixed_conditions: Dict[str, Any] = None,
        origin_dim_name: str = "origin",
        destination_dim_name: str = "destination"
    ) -> np.ndarray:
        """
        Convenience function: returns a 2D np.ndarray for O–D,
        after applying any fixed conditions on other dimensions.
        """
        if fixed_conditions is None:
            fixed_conditions = {}

        sub_tensor = self.extract_subtensor(fixed_conditions)

        # Identify the indices for the origin & destination dims
        od_dims = {origin_dim_name: None, destination_dim_name: None}
        sub_dims = sub_tensor.dims_info

        for i, d in enumerate(sub_dims):
            if d["name"] in od_dims:
                od_dims[d["name"]] = i

        if od_dims[origin_dim_name] is None or od_dims[destination_dim_name] is None:
            raise ValueError(
                f"Could not find origin/destination dimension names "
                f"('{origin_dim_name}', '{destination_dim_name}') in sub-tensor dims: "
                f"{[d['name'] for d in sub_dims]}"
            )

        o_idx = od_dims[origin_dim_name]
        d_idx = od_dims[destination_dim_name]

        # If the sub-tensor has exactly 2 dims, reorder if needed
        if sub_tensor.n_dims == 2:
            if (o_idx, d_idx) != (0, 1):
                od_matrix = np.transpose(sub_tensor.data, axes=(o_idx, d_idx))
            else:
                od_matrix = sub_tensor.data
        else:
            # If more than 2 dims remain, you need to fix them or handle them differently
            raise ValueError(
                f"After applying conditions {fixed_conditions}, sub-tensor is shape={sub_tensor.data.shape}, "
                f"not purely 2D. Remaining dims: {[d['name'] for d in sub_dims]}"
            )

        return od_matrix


# -------------------------------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic example: 3D data (time, origin, destination)
    np.random.seed(0)
    shape = (2, 3, 4)
    data_3d = np.random.randint(0, 100, size=shape)
    dims_info = [
        {"name": "time", "ids": ["t0", "t1"]},
        {"name": "origin", "ids": ["O1", "O2", "O3"]},
        {"name": "destination", "ids": ["D1", "D2", "D3", "D4"]},
    ]

    # Build and store
    tensor_obj = MultiDimTensorZarr(data_3d, dims_info)
    zarr_path = "output/travel_times_3d.zarr"
    group_name = "od_time_data"
    tensor_obj.store_to_zarr(zarr_path, group_name=group_name, overwrite=True)

    # Load
    loaded_obj = MultiDimTensorZarr.load_from_zarr(zarr_path, group_name=group_name)
    print("Loaded shape:", loaded_obj.data.shape)

    # Suppose we did a CP decomposition externally; we have factor_matrices, weights, reconstructed_data
    # We'll just create random data for demonstration
    factor_matrices_demo = [np.random.rand(s, 2) for s in shape]  # rank=2
    weights_demo = np.array([1.0, 2.0])
    reconstructed_demo = np.random.rand(*shape)
    reconstructed_dims_info = dims_info  # same shape, same dimension labels

    # Store these results
    loaded_obj.store_regression_result(
        store_path=zarr_path,
        group_name=group_name,
        result_group_name="regression_rank2",
        factor_matrices=factor_matrices_demo,
        weights=weights_demo,
        reconstructed_data=reconstructed_demo,
        reconstructed_dims_info=reconstructed_dims_info
    )
    print("Stored regression results with dimension info.")

    # Now, load the regression results back
    reg_data = MultiDimTensorZarr.load_regression_result(
        store_path=zarr_path,
        group_name=group_name,
        result_group_name="regression_rank2"
    )
    print("Loaded factor_matrices:", len(reg_data["factor_matrices"]) if reg_data["factor_matrices"] else None)
    print("Loaded weights:", reg_data["weights"])

    # If dimension info was stored for 'reconstructed_data', we can slice it as a new MultiDimTensorZarr
    rec_tensor = reg_data["reconstructed"]
    if rec_tensor is not None:
        print("Reconstructed data shape:", rec_tensor.data.shape)
        # Example: fix time="t0" and get an O-D slice
        od_matrix = rec_tensor.extract_od_slice({"time": "t0"})
        print("O–D slice from reconstructed data (time=t0): shape=", od_matrix.shape)
