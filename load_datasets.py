from datasetsforecast.m3 import M3

# The series a load as a tuple, the first element is the series themselves
def monthly_data() -> dict:
    Y_df, _, _ = M3.load(directory="./data/monthly", group="Monthly")
    unique_ids = Y_df["unique_id"].unique()[:30]

    # Putting all separated series, splitted into train and test sets, into a dictionary.

    monthly_series_dict = {
        f"{uid}": (
            Y_df[Y_df["unique_id"] == uid]
            .iloc[:-18, :]
            .drop(labels="unique_id", axis=1)
            .set_index("ds")
            .squeeze(),
            Y_df[Y_df["unique_id"] == uid]
            .iloc[-18:, :]
            .drop(labels="unique_id", axis=1)
            .set_index("ds")
            .squeeze(),
        )
        for i, uid in enumerate(unique_ids)
    }

    return monthly_series_dict
