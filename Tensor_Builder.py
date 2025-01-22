import os.path

import pandas as pd
import numpy as np
from TNT_Tensor import MultiDimTensorZarr


def build_node_id(road, direction, lat, lon):
    """
    Example: create a string that uniquely identifies a node
    by road name, direction, and lat/lon truncated to ~5 decimal places.
    You could choose another scheme if you like!
    """
    # Round lat/lon to avoid floating tiny differences
    lat_rounded = round(float(lat), 5)
    lon_rounded = round(float(lon), 5)
    # Build a simple node identifier string
    return f"{road}_{direction}_{lat_rounded}_{lon_rounded}"

def main(data_Path,output_Path):
    # -------------------------------------------------
    # 1) LOAD TMC_Identification: parse each row -> (O, D)
    # -------------------------------------------------
    tmc_ident_path = os.path.join(data_Path,"TMC_Identification.csv")  # adjust as needed
    df_tmc = pd.read_csv(tmc_ident_path)

    # We will build:
    #   tmc_map[tmc] = (origin_node_id, destination_node_id)
    # To avoid duplicates, let's maintain a dictionary lat/lon -> node_id
    node_dict = {}  # key: (lat_rounded, lon_rounded), value: node_id
    next_node_id_num = 0

    def get_or_make_node(road, direction, lat, lon):
        """Return a stable node_id from lat/lon, or create a new one if not seen before."""
        nonlocal next_node_id_num
        lat_rounded = round(float(lat), 5)
        lon_rounded = round(float(lon), 5)
        key = (lat_rounded, lon_rounded)

        if key in node_dict:
            return node_dict[key]
        else:
            # Option A: Just auto-increment
            # node_id = f"N{next_node_id_num}"
            # Option B: use road/direction in the name
            node_id = build_node_id(road, direction, lat_rounded, lon_rounded)
            node_dict[key] = node_id
            next_node_id_num += 1
            return node_id

    tmc_map = {}
    for idx, row in df_tmc.iterrows():
        tmc_code = str(row["tmc"])  # or row["tmc"] if guaranteed string

        # We'll pick columns for origin coords: (start_latitude, start_longitude)
        # and for destination coords: (end_latitude, end_longitude)
        # Also incorporate the 'road' and 'direction' for naming the node
        road = str(row["road"]) if "road" in row else "UnknownRoad"
        direction = str(row["direction"]) if "direction" in row else "UnknownDir"

        start_lat = row["start_latitude"]
        start_lon = row["start_longitude"]
        end_lat = row["end_latitude"]
        end_lon = row["end_longitude"]

        origin_node = get_or_make_node(road, direction, start_lat, start_lon)
        destination_node = get_or_make_node(road, direction, end_lat, end_lon)

        tmc_map[tmc_code] = (origin_node, destination_node)

    # -------------------------------------------------
    # 2) LOAD Reading.csv: each row has tmc_code, measurement_tstamp, travel_time_seconds, etc.
    # -------------------------------------------------
    reading_path =os.path.join(data_Path,"Reading.csv" )  # adjust as needed
    df_read = pd.read_csv(reading_path)

    # Convert measurement_tstamp to a standard datetime object
    # The sample is something like "1/16/2025 0:00"
    # We'll parse with a suitable format or rely on pandas auto-infer:
    df_read["measurement_tstamp"] = pd.to_datetime(df_read["measurement_tstamp"])

    # Filter out any rows with TMC not found in tmc_map (just in case)
    df_read = df_read[df_read["tmc_code"].isin(tmc_map.keys())].copy()

    # For convenience, let's add columns for origin_node, destination_node
    # by mapping tmc_code -> (O, D)
    origins = []
    destinations = []
    for tmc_code in df_read["tmc_code"]:
        O, D = tmc_map[tmc_code]
        origins.append(O)
        destinations.append(D)

    df_read["origin_node"] = origins
    df_read["destination_node"] = destinations

    # -------------------------------------------------
    # 3) Build a 3D data structure: (time, O, D) or (O, D, time)
    #    For demonstration, let's define "time" in hours (year-mon-day-hour).
    # -------------------------------------------------

    # We'll add columns year, month, day, hour to facilitate grouping
    df_read["year"] = df_read["measurement_tstamp"].dt.year
    df_read["month"] = df_read["measurement_tstamp"].dt.month
    df_read["day"] = df_read["measurement_tstamp"].dt.day
    df_read["hour"] = df_read["measurement_tstamp"].dt.hour

    # For example, define "time_key" as "YYYY-MM-DD HH:00"
    df_read["time_key"] = df_read["measurement_tstamp"].dt.strftime("%Y-%m-%d %H:00")

    # We'll group by (origin_node, destination_node, time_key) and average the travel_time_seconds
    group_cols = ["origin_node", "destination_node", "time_key"]
    grouped = df_read.groupby(group_cols)["travel_time_seconds"].mean().reset_index()

    # Now we have rows: [origin_node, destination_node, time_key, average_travel_time]
    # We want a 3D array with dimension order, say, time, O, D.
    # We'll get unique values for each dimension:
    unique_times = grouped["time_key"].unique()
    unique_origins = grouped["origin_node"].unique()
    unique_destinations = grouped["destination_node"].unique()

    # Sort them for consistent ordering
    unique_times = np.sort(unique_times)
    unique_origins = np.sort(unique_origins)
    unique_destinations = np.sort(unique_destinations)

    # Build an index mapping to array indices
    time_index_map = {t: i for i, t in enumerate(unique_times)}
    origin_index_map = {o: i for i, o in enumerate(unique_origins)}
    dest_index_map = {d: i for i, d in enumerate(unique_destinations)}

    # Create an empty 3D array: shape = (len(unique_times), len(unique_origins), len(unique_destinations))
    # We'll fill with np.nan initially
    data_3d = np.full(
        shape=(len(unique_times), len(unique_origins), len(unique_destinations)),
        fill_value=np.nan,
        dtype=float
    )

    # Populate
    for i, row in grouped.iterrows():
        O = row["origin_node"]
        D = row["destination_node"]
        T = row["time_key"]
        avg_tt = row["travel_time_seconds"]

        t_idx = time_index_map[T]
        o_idx = origin_index_map[O]
        d_idx = dest_index_map[D]
        data_3d[t_idx, o_idx, d_idx] = avg_tt

    # Create dims_info
    # dimension 0 -> "time" with IDs = unique_times
    # dimension 1 -> "O" with IDs = unique_origins
    # dimension 2 -> "D" with IDs = unique_destinations
    dims_info = [
        {
            "name": "time",
            "ids": unique_times.tolist(),
        },
        {
            "name": "O",
            "ids": unique_origins.tolist(),
        },
        {
            "name": "D",
            "ids": unique_destinations.tolist(),
        },
    ]

    # Instantiate and store by Zarr
    tensor_3d = MultiDimTensorZarr(data_3d, dims_info)
    zarr_path = os.path.join(output_Path,"travel_times_3d.zarr")
    group_name = "od_time_data"
    tensor_3d.store_to_zarr(zarr_path, group_name=group_name, overwrite=True)
    print(f"Stored 3D travel-time data to: {zarr_path}")

    # -------------------------------------------------
    # 4) Implement a "powerful" filter function:
    #    - Takes year, month, day, hour
    #    - Returns a 2D (O, D) average
    # -------------------------------------------------
    # We'll do something like:
    #   1) filter df_read by the given date/time
    #   2) group by O,D
    #   3) average travel_time_seconds
    #   4) store a new 2D array in a Zarr if desired

    def filter_and_store(year=None, month=None, day=None, hour=None, out_zarr="travel_times_filtered_2d.zarr"):
        # Filter
        filtered = df_read.copy()
        if year is not None:
            filtered = filtered[filtered["year"] == year]
        if month is not None:
            filtered = filtered[filtered["month"] == month]
        if day is not None:
            filtered = filtered[filtered["day"] == day]
        if hour is not None:
            filtered = filtered[filtered["hour"] == hour]

        if filtered.empty:
            print("No rows found for given filter.")
            return

        # Group by (O, D), average travel_time_seconds
        grouped_2d = filtered.groupby(["origin_node", "destination_node"])["travel_time_seconds"].mean().reset_index()

        # Build unique O, D
        Os = np.sort(grouped_2d["origin_node"].unique())
        Ds = np.sort(grouped_2d["destination_node"].unique())

        # Create mapping
        o_index_map = {o: i for i, o in enumerate(Os)}
        d_index_map = {d: i for i, d in enumerate(Ds)}

        # Create 2D array
        data_2d = np.full((len(Os), len(Ds)), fill_value=np.nan, dtype=float)

        for _, rowx in grouped_2d.iterrows():
            Ox = rowx["origin_node"]
            Dx = rowx["destination_node"]
            avg_tt = rowx["travel_time_seconds"]
            data_2d[o_index_map[Ox], d_index_map[Dx]] = avg_tt

        # dims_info for 2D
        dims_info_2d = [
            {"name": "O", "ids": Os.tolist()},
            {"name": "D", "ids": Ds.tolist()},
        ]

        # Instantiate and store
        tensor_2d = MultiDimTensorZarr(data_2d, dims_info_2d)

        # We'll store group_name as something descriptive
        # e.g. "filtered_year2025_month1_day16_hour0"
        group_name_2d = f"filtered_y{year}_m{month}_d{day}_h{hour}"
        tensor_2d.store_to_zarr(out_zarr, group_name=group_name_2d, overwrite=False)
        print(f"Filtered 2D data stored to {out_zarr} group={group_name_2d}")

    # -------------------------------------------------
    # 5) Example usage of filter:
    #    Suppose we want data for year=2025, month=1, day=16, hour=0
    # -------------------------------------------------
    filter_and_store(year=2025, month=1, day=16, hour=10,out_zarr=os.path.join(output_Path,"travel_times_2d.zarr")  )

if __name__ == "__main__":
    main("data/Tempe","output")
