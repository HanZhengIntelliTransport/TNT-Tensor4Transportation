"""
Example usage of the MultiDimTensorZarr class to:
1) Load a high-dimensional travel-time tensor from Zarr.
2) Fix certain dimensions (excluding O and D).
3) Perform a CP decomposition to approximate the data.
4) Extract or reconstruct an O–D 2D slice under certain 'conditions'.
5) Generate lower-rank approximations of that 2D slice.
"""

import numpy as np
# If you want to do advanced tensor decomposition, you might need:
# pip install tensorly
# Then import:
import tensorly as tl
from tensorly.decomposition import parafac

# Import the class from your TNT_Tensor.py
from TNT_Tensor import MultiDimTensorZarr


def load_high_dim_tensor(zarr_path: str, group_name: str) -> MultiDimTensorZarr:
    """
    Load the high-dimensional tensor from a Zarr store using MultiDimTensorZarr.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr directory.
    group_name : str
        Group name in the Zarr store.

    Returns
    -------
    MultiDimTensorZarr
        The loaded tensor object.
    """
    return MultiDimTensorZarr.load_from_zarr(zarr_path, group_name)


def slice_excluding_OD(tensor_obj: MultiDimTensorZarr, fixed_conditions: dict):
    """
    Given a high-dimensional tensor that includes "origin" and "destination"
    as two of its dimensions, fix certain other dimension(s) according to 'fixed_conditions'.

    For example, if your dimension names are:
        ["time", "day_type", "origin", "destination", "something_else"]
    and you pass fixed_conditions = {"time": "t0", "day_type": "weekend"},
    this function will slice the tensor where time="t0" and day_type="weekend",
    leaving a sub-tensor whose shape is just (origin, destination, [any other unfixed dims]).

    This is just a helper to isolate the data we need for O–D analysis
    at specific values of other dimensions.

    Parameters
    ----------
    tensor_obj : MultiDimTensorZarr
        The loaded high-dimensional tensor.
    fixed_conditions : dict
        Dictionary of { dimension_name: dimension_value } that you want to fix.

    Returns
    -------
    sub_data : np.ndarray
        The sliced array that excludes the fixed dimensions (i.e., they've
        been indexed down to single values).
    sub_dims_info : list of dict
        The updated dims_info after removing the fixed dimensions.
    """
    data = tensor_obj.data
    dims_info = tensor_obj.dims_info

    # We'll keep track of the indices we want to select for each dimension.
    # If a dimension is in fixed_conditions, we pick the single matching index.
    # Otherwise, we keep it as a slice (i.e., all).
    index_slices = [slice(None)] * tensor_obj.n_dims  # start with all slices

    # We'll build new dims_info for the remaining dimensions
    new_dims_info = []

    for dim_idx, dim_dict in enumerate(dims_info):
        dim_name = dim_dict["name"]
        dim_ids = np.asarray(dim_dict["ids"])

        if dim_name in fixed_conditions:
            # We want to fix this dimension to a single value
            desired_val = fixed_conditions[dim_name]
            # find the index of that desired_val
            matches = np.where(dim_ids == desired_val)[0]
            if len(matches) == 0:
                raise ValueError(f"No match for value {desired_val} in dimension '{dim_name}'.")
            match_idx = matches[0]
            # fix this dimension
            index_slices[dim_idx] = match_idx
            # we do NOT add this dimension to new_dims_info because it's now "collapsed"
        else:
            # we keep this dimension in the sub-tensor
            new_dims_info.append({
                "name": dim_name,
                "ids": dim_ids
            })

    # Use NumPy advanced slicing
    sub_data = data[tuple(index_slices)]

    # If sub_data is still multi-dimensional but with fewer axes,
    # it might have shape e.g. (len(O), len(D)) or maybe more, depending on leftover dims.
    # new_dims_info has only the "free" dimensions.

    return sub_data, new_dims_info


def cp_decompose_and_reconstruct(
        data: np.ndarray, rank: int
) -> np.ndarray:
    """
    Perform a simple CP decomposition of the given multi-dimensional data
    and then reconstruct it with the specified rank.

    Note: This uses TensorLy (parafac) as an example of "tensor regression" or decomposition.
    In practice, you might use advanced regression or factorization methods.

    Parameters
    ----------
    data : np.ndarray
        The array to decompose (could be 2D, 3D, 4D, etc.).
    rank : int
        The rank for CP decomposition.

    Returns
    -------
    reconstructed : np.ndarray
        The data reconstructed from the CP decomposition factors at the given rank.
    """
    # Convert data to float if it's not already
    data_float = data.astype(float)

    # Perform CP decomposition (Parafac)
    # factor_matrices is a list of arrays, each corresponding to a mode of `data`
    weights, factor_matrices = parafac(data_float, rank=rank, init='random', tol=1e-6, n_iter_max=200)

    # Reconstruct from the decomposition
    reconstructed = tl.kruskal_to_tensor((weights, factor_matrices))
    return reconstructed


def create_2d_slices_for_various_ranks(
        data_2d: np.ndarray,
        ranks: list
) -> dict:
    """
    Given a 2D array (e.g., an O–D matrix), decompose it at various ranks
    and return a dictionary of reconstructed 2D slices.

    Parameters
    ----------
    data_2d : np.ndarray
        A 2D array, shape = (n_origins, n_destinations), or (some_rows, some_cols).
    ranks : list
        List of integer ranks to attempt.

    Returns
    -------
    reconstructions : dict
        A dict of { rank_value: 2D_reconstructed_array }.
    """
    reconstructions = {}
    # 2D "CP decomposition" is effectively the same as a low-rank factorization
    for r in ranks:
        # We'll do a CP decomposition on a 2D matrix.
        # This is somewhat analogous to an SVD but using the same TensorLy method.
        rec = cp_decompose_and_reconstruct(data_2d, rank=r)
        reconstructions[r] = rec
    return reconstructions


def main(zarr_path = "output/travel_times_3d.zarr",group_name = "od_time_data"):
    # -----------------------------------------------------------------------
    # 1) Load the high-dimensional OD-time data
    # -----------------------------------------------------------------------
    od_tensor = MultiDimTensorZarr.load_from_zarr(zarr_path, group_name)
    print("Loaded tensor shape:", od_tensor.data.shape)
    for i in range(od_tensor.n_dims):
        print("Dimension", i, od_tensor.get_dim_info(i))

    # Suppose the dimension names are something like:
    #   "time", "day_type", "origin", "destination"
    # or possibly more. We want to fix "time" and "day_type" to certain values
    # and keep "origin" and "destination" free so we get a 2D O–D matrix.

    # -----------------------------------------------------------------------
    # 2) Slice the high-dimensional data to fix certain dimensions
    # -----------------------------------------------------------------------
    # For example, let's assume we fix time="2025-01-16 00:00"
    # and day_type="weekend" (just placeholders).
    # Adjust dimension names/values to match your actual data.
    fixed_conditions = {
        "time": "2025-01-16 00:00",
        "day_type": "weekend",
    }

    # We'll get the sub-data plus the new dimension info
    sub_data, sub_dims_info = slice_excluding_OD(od_tensor, fixed_conditions)
    # sub_data might be shape (num_origins, num_destinations) if
    #   origin and destination are the only leftover dimensions.

    print("Sliced data shape:", sub_data.shape)
    print("Sliced dims_info:", sub_dims_info)

    # Check if it's really 2D. If there's more leftover dims,
    # you might need to fix them in `fixed_conditions` or handle them differently.
    if sub_data.ndim != 2:
        print("WARNING: Sliced data is not 2D. It's shape:", sub_data.shape)
        print("We'll proceed, but consider adjusting the fixed_conditions further.")

    # -----------------------------------------------------------------------
    # 3) Perform "tensor regression"/CP decomposition on the 2D slice
    #    for different ranks. We'll create a few rank approximations.
    # -----------------------------------------------------------------------
    ranks_to_try = [1, 2, 3]  # e.g., low rank, moderate rank
    recon_map = create_2d_slices_for_various_ranks(sub_data, ranks=ranks_to_try)

    # Now we have approximations of the O–D matrix at different ranks.
    for r in ranks_to_try:
        print(f"\nRank={r} Reconstruction shape:", recon_map[r].shape)
        # If you want to store each reconstructed matrix as a new Zarr group, you can do so.
        # For demonstration, we'll just show how you might do it:

        # Make dims_info for this 2D approximation
        # We'll reuse sub_dims_info for dimension names/IDs.
        # It's presumably something like: [ {"name": "origin", "ids": [...]}, {"name": "destination", "ids": [...]} ]
        approx_dims_info = sub_dims_info.copy()  # shallow copy is fine if we don't modify it

        approx_tensor_obj = MultiDimTensorZarr(data=recon_map[r], dims_info=approx_dims_info)
        # If you want to store it:
        approx_zarr_path = "output/travel_times_approx.zarr"
        approx_group_name = f"rank_{r}_slice"
        approx_tensor_obj.store_to_zarr(
            store_path=approx_zarr_path,
            group_name=approx_group_name,
            overwrite=False  # or True if you want to overwrite
        )
        print(f"Stored rank-{r} O–D approximation to Zarr group: {approx_group_name}")

    # -----------------------------------------------------------------------
    # DONE: we have demonstrated:
    #  1) loading the high-dimensional data
    #  2) slicing out the non-OD dimensions
    #  3) applying a CP decomposition (as a simple example of 'tensor regression')
    #  4) storing the 2D O–D approximations of various ranks
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    zarr_path = "output/travel_times_3d.zarr"
    group_name = "od_time_data"
    main(zarr_path,group_name)