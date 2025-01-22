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
    # NEW / UPDATED METHODS: For Tensor Regression Results
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
        """
        import zarr  # local import if you prefer
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
                    del fm_group[ds_name]
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
            if "data" in rec_group:
                del rec_group["data"]
            rec_array = rec_group.empty(
                name="data",
                shape=reconstructed_data.shape,
                dtype=reconstructed_data.dtype,
            )
            rec_array[:] = reconstructed_data

            # If dims info is provided, store it
            if reconstructed_dims_info is not None:
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
        """
        import zarr
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
                # Check if we have dims_info
                dims_info_attr = rec_group.attrs.get("dims_info", None)
                if dims_info_attr is not None:
                    rec_dims_info = []
                    for dim_dict in dims_info_attr:
                        rec_dims_info.append({
                            "name": dim_dict["name"],
                            "ids": dim_dict["ids"]
                        })
                    rec_tensor = cls(rec_data, rec_dims_info)
                    out_dict["reconstructed"] = rec_tensor
                else:
                    # If no dims_info, we can't reconstruct a MultiDimTensorZarr easily
                    # You could store raw rec_data in out_dict if desired
                    pass

        return out_dict

    # -------------------------------------------------------------------------
    # Slicing / sub-tensor extraction
    # -------------------------------------------------------------------------
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

        # Identify the indices for origin & destination
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

        if sub_tensor.n_dims == 2:
            # If sub-tensor is exactly 2D, reorder if needed
            if (o_idx, d_idx) != (0, 1):
                od_matrix = np.transpose(sub_tensor.data, axes=(o_idx, d_idx))
            else:
                od_matrix = sub_tensor.data
        else:
            raise ValueError(
                f"After applying conditions {fixed_conditions}, sub-tensor shape={sub_tensor.data.shape}, "
                f"not purely 2D. Dimensions: {[d['name'] for d in sub_dims]}"
            )

        return od_matrix

    def extract_od_slice_to_tensor(
        self,
        fixed_conditions: Dict[str, Any] = None,
        origin_dim_name: str = "origin",
        destination_dim_name: str = "destination"
    ) -> "MultiDimTensorZarr":
        """
        Similar to extract_od_slice(...), but returns a new MultiDimTensorZarr
        of shape (num_origins, num_destinations), preserving dimension info
        so you can further store or manipulate it as a tensor object.

        Returns
        -------
        MultiDimTensorZarr
            A 2D MultiDimTensorZarr with dims = [origin, destination].
        """
        if fixed_conditions is None:
            fixed_conditions = {}

        # First, create a sub-tensor with everything except time/other fixed dims
        sub_tensor = self.extract_subtensor(fixed_conditions)

        # Identify origin/destination dimension indices
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

        # Check shape
        if sub_tensor.n_dims != 2:
            raise ValueError(
                f"After applying conditions {fixed_conditions}, we do not have a 2D sub-tensor."
                f" shape={sub_tensor.data.shape}, dims={[d['name'] for d in sub_dims]}"
            )

        # Reorder axes if necessary so that origin is axis=0, destination axis=1
        od_data = sub_tensor.data
        if (o_idx, d_idx) != (0, 1):
            od_data = np.transpose(sub_tensor.data, axes=(o_idx, d_idx))

        # Build new dims_info in correct axis order
        origin_info = sub_dims[o_idx]
        dest_info = sub_dims[d_idx]
        new_dims_info = [
            {"name": origin_info["name"], "ids": origin_info["ids"]},
            {"name": dest_info["name"],  "ids": dest_info["ids"]}
        ]

        return MultiDimTensorZarr(od_data, new_dims_info)

    # -------------------------------------------------------------------------
    # OPTIONAL HELPER: Extract partial factor matrices when slicing
    # -------------------------------------------------------------------------
    def extract_sub_factors(
        self,
        factor_matrices: List[np.ndarray],
        fixed_conditions: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        If you have a CP decomposition with factor_matrices for each dimension
        (e.g. factor_matrices[d] has shape=(size_of_dim_d, rank)), this method
        returns a new list of partial factor matrices that correspond to the
        sub-tensor after applying 'fixed_conditions'.

        For each fixed dimension d, we select exactly one row from factor_matrices[d].
        For each free dimension, we keep the entire factor_matrices[d].

        This is a naive approach that can be useful if you want to interpret
        the factor for a single time slice or a single day_type, etc.

        Returns
        -------
        sub_factors : List[np.ndarray]
            The partial factor matrices. For each fixed dimension, shape will be (1, rank).
            For each free dimension, shape remains (size_of_dim_d, rank).
        """
        # We'll rely on the dimension order being the same as factor_matrices order:
        # dimension i -> factor_matrices[i], shape = (len(dims_info[i]["ids"]), rank).
        # If your factor_matrices are in a different order, you'll need to adapt.
        sub_factors = []
        rank = factor_matrices[0].shape[1] if factor_matrices else 0

        for dim_idx, dim_dict in enumerate(self.dims_info):
            dim_name = dim_dict["name"]
            dim_ids = np.asarray(dim_dict["ids"])
            fm = factor_matrices[dim_idx]  # shape=(dim_size, rank)
            if fm.shape[0] != len(dim_ids):
                raise ValueError(f"Factor matrix {dim_idx} does not match dimension size for '{dim_name}'")

            if dim_name in fixed_conditions:
                # pick exactly the row for the matched index
                desired_val = fixed_conditions[dim_name]
                matches = np.where(dim_ids == desired_val)[0]
                if len(matches) == 0:
                    raise ValueError(f"No match for {desired_val} in dimension '{dim_name}'.")
                idx_val = matches[0]
                sub_fm = fm[idx_val:idx_val+1, :]  # shape=(1, rank)
            else:
                # keep entire factor
                sub_fm = fm

            sub_factors.append(sub_fm)
        return sub_factors


# ------------------------------------------------------------------------------
# DEMO USAGE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Let's say we have a 3D tensor: (time, origin, destination)
    np.random.seed(0)
    data_3d = np.random.randint(0, 100, (2, 3, 4))
    dims_info_3d = [
        {"name": "time", "ids": ["t0", "t1"]},
        {"name": "origin", "ids": ["O1", "O2", "O3"]},
        {"name": "destination", "ids": ["D1", "D2", "D3", "D4"]}
    ]
    tensor_3d = MultiDimTensorZarr(data_3d, dims_info_3d)

    # Suppose we performed a CP decomposition externally:
    # factor_matrices: [time_factor, origin_factor, dest_factor]
    # shape: time_factor=(2, rank), origin_factor=(3, rank), dest_factor=(4, rank)
    rank = 2
    factor_matrices_demo = [
        np.random.rand(2, rank),  # time dimension
        np.random.rand(3, rank),  # origin dimension
        np.random.rand(4, rank),  # destination dimension
    ]
    # We'll store them, plus a "reconstructed_data"
    reconstructed_data = np.random.rand(2, 3, 4)  # same shape
    zarr_path = "output/demo_3d.zarr"
    group_name = "od_time_data"

    # Save original data
    tensor_3d.store_to_zarr(zarr_path, group_name, overwrite=True)

    # Then store the regression results
    tensor_3d.store_regression_result(
        store_path=zarr_path,
        group_name=group_name,
        result_group_name="cp_rank2_demo",
        factor_matrices=factor_matrices_demo,
        weights=None,
        reconstructed_data=reconstructed_data,
        reconstructed_dims_info=dims_info_3d  # so we can slice the reconstruction
    )
    print("Stored CP rank-2 results.")

    # Load them back
    reg_results = MultiDimTensorZarr.load_regression_result(
        store_path=zarr_path,
        group_name=group_name,
        result_group_name="cp_rank2_demo"
    )
    print("Loaded factor_matrices count:", len(reg_results["factor_matrices"]) if reg_results["factor_matrices"] else None)

    rec_tensor = reg_results["reconstructed"]
    if rec_tensor is not None:
        print("Reconstructed data shape:", rec_tensor.data.shape)
        # For example, fix time="t0" and get O–D sub-tensor as a 2D array
        od_2d_array = rec_tensor.extract_od_slice({"time": "t0"})
        print("O–D array shape (time=t0):", od_2d_array.shape)

        # Or get a new MultiDimTensorZarr for that O–D slice
        od_tensor = rec_tensor.extract_od_slice_to_tensor({"time": "t0"})
        print("od_tensor shape:", od_tensor.data.shape)
        print("od_tensor dims_info:", od_tensor.dims_info)

    # Suppose we also want the partial factor matrices that correspond to time="t0":
    sub_factors = tensor_3d.extract_sub_factors(reg_results["factor_matrices"], {"time": "t0"})
    print("Partial factor for time dimension (should be shape (1, rank)):", sub_factors[0].shape)
    print("Factor for origin dimension (should remain (3, rank)):", sub_factors[1].shape)
    print("Factor for destination dimension (should remain (4, rank)):", sub_factors[2].shape)