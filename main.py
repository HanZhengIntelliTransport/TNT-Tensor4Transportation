from TNT_Tensor import MultiDimTensorZarr


zarr_path = "output/travel_times_3d.zarr"
group_name = "od_time_data"
od_tensor = MultiDimTensorZarr.load_from_zarr(zarr_path, group_name=group_name)
print("Loaded shape:", od_tensor.data.shape)

    # Suppose we did a CP decomposition externally; we have factor_matrices, weights, reconstructed_data
    # We'll just create random data for demonstration
    factor_matrices_demo = [np.random.rand(s, 2) for s in shape]  # rank=2
    weights_demo = np.array([1.0, 2.0])
    reconstructed_demo = np.random.rand(*shape)
    reconstructed_dims_info = dims_info  # same shape, same dimension labels

    # Store these results
    od_tensor.store_regression_result(
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


