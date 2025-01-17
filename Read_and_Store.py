# read_and_store.py
import pandas as pd
import numpy as np
from TNT_Tensor import MultiDimTensorZarr


def main():
    # 1. Read the CSV file
    csv_file = "data/I-95/Travel_Time/OD_All.csv"  # replace with your actual filename/path
    df = pd.read_csv(csv_file)

    # 2. Make sure "Origin Zone Name", "Destination Zone Name", and "Avg Travel Time (sec)"
    #    columns exist and are spelled exactly as in the CSV.
    #    If necessary, rename columns to a simpler format:
    # df.rename(columns={
    #     "Origin Zone Name": "origin_name",
    #     "Destination Zone Name": "dest_name",
    #     "Avg Travel Time (sec)": "avg_tt_sec"
    # }, inplace=True)
    #
    # Then pivot. But for demonstration, let's assume the columns are the same as
    # in your data. If not, adjust the pivot call.

    # 3. Pivot so that each row is a unique Origin, each column is a unique Destination,
    #    and the cells are average travel time (sec).
    pivot_df = df.pivot(
        index="Origin Zone Name",
        columns="Destination Zone Name",
        values="Avg Travel Time (sec)"
    )

    # 4. If there are any missing values, decide how to handle them:
    #    e.g., fill with 0, fill with np.nan, or drop them
    pivot_df = pivot_df.fillna(np.nan)

    # 5. Convert the pivot table to a NumPy array
    data_2d = pivot_df.to_numpy()

    # 6. Create the dims_info for the 2D data:
    #    dimension 0 = Origin (O), dimension 1 = Destination (D)
    dims_info = [
        {
            "name": "O",  # dimension name
            "ids": pivot_df.index.tolist()  # all unique origin zone names
        },
        {
            "name": "D",  # dimension name
            "ids": pivot_df.columns.tolist()  # all unique destination zone names
        },
    ]

    # 7. Instantiate the MultiDimTensorZarr object
    travel_times_tensor = MultiDimTensorZarr(
        data=data_2d,
        dims_info=dims_info
    )

    # 8. Store to Zarr
    zarr_path = "travel_time.zarr"
    group_name = "travel_times_2d"

    travel_times_tensor.store_to_zarr(
        store_path=zarr_path,
        group_name=group_name,
        overwrite=True
    )

    print("Successfully stored travel times into Zarr at:", zarr_path)

    # 9. (Optional) Load it back to confirm
    loaded_tensor = MultiDimTensorZarr.load_from_zarr(
        store_path=zarr_path,
        group_name=group_name
    )

    # 10. Print out a summary
    print("Loaded shape:", loaded_tensor.data.shape)
    print("Dimension 0 info (Origins):", loaded_tensor.get_dim_info(0))
    print("Dimension 1 info (Destinations):", loaded_tensor.get_dim_info(1))

    # 11. Optional: Retrieve a single value
    if loaded_tensor.data.size > 0:
        val, dim_labels = loaded_tensor.get_value(0, 0)  # top-left element
        print("Example cell value (row=0, col=0):", val)
        print("Corresponding O, D labels:", dim_labels)


if __name__ == "__main__":
    main()
