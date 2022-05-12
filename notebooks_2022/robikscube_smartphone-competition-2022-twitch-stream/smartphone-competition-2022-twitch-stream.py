#!/usr/bin/env python
# coding: utf-8

# # Google Smartphone Decimeter Challenge 2022
# ![myimg](https://thehackernews.com/images/-MWrikkgPuOk/Wk4SJfLxqAI/AAAAAAAAvYw/sPCXIrCBxvQfuUBFg-v_yJvD1wllXdTcgCLcBGAs/s728-e100/gps-location-tracking-device.png)
# 
# This notebook was created during a live stream on twitch:
# - Check out my twitch channel here: [link](https://www.twitch.tv/medallionstallion_)
# - Shameless plug for my youtube channel with videos about data science and machine learning. [Check it out here.](https://www.youtube.com/channel/UCxladMszXan-jfgzyeIMyvw)

# # Data
# 
# In this competition we are tasked with identifying the exact location of a phone using it's GPS data. As the competition description says they would like us to find the... "location down to decimeter or even centimeter resolution which could enable services that require lane-level accuracy such as HOV lane ETA estimation."
# 

# In[ ]:


get_ipython().system('pip install nb_black > /dev/null')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'lab_black')


# In[ ]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly.express as px

pd.set_option("max_columns", 500)


# ## Training Data
# 
# The data is setup in the `smartphone-decimeter-2022` folder into train and test. Each route is it's own folder like `2020-05-15-US-MTV-1` each phone is a folder under the route like `GooglePixel4XL`.
# 
# The main data sources are:
# - The target: For each phone there is a `ground_truth.csv` with it's location at timestamps.
# - Training data `device_gnss.csv` Each row contains raw GNSS measurements.
# - Training data `device_imu.csv` Readings the phone's accelerometer, gyroscope, and magnetometer. 
# 

# In[ ]:


trip_id = "2020-05-15-US-MTV-1/GooglePixel4XL"


# In[ ]:


# Pull Data for an example phone
gt = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/ground_truth.csv"
)
gnss = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/device_gnss.csv"
)
imu = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/device_imu.csv"
)


# # Create Baseline Submission using GNSS Data
# Shout out to this notebook for the helper code that will get us started converting GNSS to lat/long
# 
# https://www.kaggle.com/code/saitodevel01/gsdc2-baseline-submission

# In[ ]:


import glob
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

INPUT_PATH = "../input/smartphone-decimeter-2022"

WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3

HAVERSINE_RADIUS = 6_371_000


@dataclass
class ECEF:
    x: np.array
    y: np.array
    z: np.array

    def to_numpy(self):
        return np.stack([self.x, self.y, self.z], axis=0)

    @staticmethod
    def from_numpy(pos):
        x, y, z = [np.squeeze(w) for w in np.split(pos, 3, axis=-1)]
        return ECEF(x=x, y=y, z=z)


@dataclass
class BLH:
    lat: np.array
    lng: np.array
    hgt: np.array


def ECEF_to_BLH(ecef):
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    e2 = WGS84_SQUARED_FIRST_ECCENTRICITY
    e2_ = WGS84_SQUARED_SECOND_ECCENTRICITY
    x = ecef.x
    y = ecef.y
    z = ecef.z
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(z * (a / b), r)
    B = np.arctan2(z + (e2_ * b) * np.sin(t) ** 3, r - (e2 * a) * np.cos(t) ** 3)
    L = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2 * np.sin(B) ** 2)
    H = (r / np.cos(B)) - n
    return BLH(lat=B, lng=L, hgt=H)


def haversine_distance(blh_1, blh_2):
    dlat = blh_2.lat - blh_1.lat
    dlng = blh_2.lng - blh_1.lng
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(blh_1.lat) * np.cos(blh_2.lat) * np.sin(dlng / 2) ** 2
    )
    dist = 2 * HAVERSINE_RADIUS * np.arcsin(np.sqrt(a))
    return dist


def pandas_haversine_distance(df1, df2):
    blh1 = BLH(
        lat=np.deg2rad(df1["LatitudeDegrees"].to_numpy()),
        lng=np.deg2rad(df1["LongitudeDegrees"].to_numpy()),
        hgt=0,
    )
    blh2 = BLH(
        lat=np.deg2rad(df2["LatitudeDegrees"].to_numpy()),
        lng=np.deg2rad(df2["LongitudeDegrees"].to_numpy()),
        hgt=0,
    )
    return haversine_distance(blh1, blh2)


def ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis):
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    columns = ["utcTimeMillis"] + ecef_columns
    ecef_df = (
        gnss_df.drop_duplicates(subset="utcTimeMillis")[columns]
        .dropna()
        .reset_index(drop=True)
    )
    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh = ECEF_to_BLH(ecef)

    TIME = ecef_df["utcTimeMillis"].to_numpy()
    lat = InterpolatedUnivariateSpline(TIME, blh.lat, ext=3)(UnixTimeMillis)
    lng = InterpolatedUnivariateSpline(TIME, blh.lng, ext=3)(UnixTimeMillis)
    return pd.DataFrame(
        {
            "tripId": tripID,
            "UnixTimeMillis": UnixTimeMillis,
            "LatitudeDegrees": np.degrees(lat),
            "LongitudeDegrees": np.degrees(lng),
        }
    )


def calc_score(tripID, pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])
    return score


# In[ ]:


ss = pd.read_csv("../input/smartphone-decimeter-2022/sample_submission.csv")


# In[ ]:


trip_id = "2020-05-15-US-MTV-1/GooglePixel4XL"
baseline = ecef_to_lat_lng(trip_id, gnss, gt["UnixTimeMillis"].values)


# # Plotting Function for the Paths

# In[ ]:


def visualize_traffic(
    df,
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    center=None,
    color_col="phone",
    label_col="tripId",
    zoom=9,
    opacity=1,
):
    if center is None:
        center = {
            "lat": df[lat_col].mean(),
            "lon": df[lon_col].mean(),
        }
    fig = px.scatter_mapbox(
        df,
        # Here, plotly gets, (x,y) coordinates
        lat=lat_col,
        lon=lon_col,
        # Here, plotly detects color of series
        color=color_col,
        labels=label_col,
        zoom=zoom,
        center=center,
        height=600,
        width=800,
        opacity=0.5,
    )
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()


def plot_gt_vs_baseline(tripId):
    """
    Create a plot of the baseline predictions vs. the ground truth
    for a given tripId
    """
    # Pull Data for an example phone
    gt = pd.read_csv(
        f"../input/smartphone-decimeter-2022/train/{tripId}/ground_truth.csv"
    )
    gnss = pd.read_csv(
        f"../input/smartphone-decimeter-2022/train/{tripId}/device_gnss.csv"
    )
    imu = pd.read_csv(
        f"../input/smartphone-decimeter-2022/train/{tripId}/device_imu.csv"
    )
    baseline = ecef_to_lat_lng(trip_id, gnss, gt["UnixTimeMillis"].values)
    # Combine ground truth with baseline predictions
    baseline["isGT"] = False
    gt["isGT"] = True
    gt["tripId"] = tripId

    combined = (
        pd.concat([baseline, gt[baseline.columns]], axis=0)
        .reset_index(drop=True)
        .copy()
    )

    # Plotting the route
    visualize_traffic(
        combined,
        lat_col="LatitudeDegrees",
        lon_col="LongitudeDegrees",
        color_col="isGT",
        zoom=10,
    )


# ## Concat the Baseline Predictions with the Ground Truth
# - If you zoom in you can see how the blue and red dots are different.
# - The baseline predictions are very noisy when the car is moving at slow speeds or stopped.

# In[ ]:


plot_gt_vs_baseline(trip_id)


# # Plot an Example and Zoom at noisy area

# In[ ]:


from glob import glob

train_gts = glob("../input/smartphone-decimeter-2022/train/*/*/ground_truth.csv")
trip_ids = ["/".join(p.split("/")[-3:-1]) for p in train_gts]


# In[ ]:


tripId = trip_ids[10]
# Pull Data for an example phone
gt = pd.read_csv(f"../input/smartphone-decimeter-2022/train/{tripId}/ground_truth.csv")
gnss = pd.read_csv(f"../input/smartphone-decimeter-2022/train/{tripId}/device_gnss.csv")
imu = pd.read_csv(f"../input/smartphone-decimeter-2022/train/{tripId}/device_imu.csv")
baseline = ecef_to_lat_lng(trip_id, gnss, gt["UnixTimeMillis"].values)
# Combine ground truth with baseline predictions
baseline["isGT"] = False
gt["isGT"] = True
gt["tripId"] = tripId

combined = (
    pd.concat([baseline, gt[baseline.columns]], axis=0).reset_index(drop=True).copy()
)


# ## Example of baseline noise
# - Note that the blue (baseline predictions) are very noisy at the start of this trip

# In[ ]:


offset = 150_000
start = 1607640760432 + offset
combined.query("UnixTimeMillis < @start")

visualize_traffic(combined.query("UnixTimeMillis < @start"), color_col="isGT", zoom=16)


# # Create Baselines for Training Set and Test Set
# 
# Using the awesome code from: https://www.kaggle.com/code/saitodevel01/gsdc2-baseline-submission

# In[ ]:


import glob

INPUT_PATH = "../input/smartphone-decimeter-2022"

sample_df = pd.read_csv(f"{INPUT_PATH}/sample_submission.csv")
pred_dfs = []
for dirname in tqdm(sorted(glob.glob(f"{INPUT_PATH}/test/*/*"))):
    drive, phone = dirname.split("/")[-2:]
    tripID = f"{drive}/{phone}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv")
    UnixTimeMillis = sample_df[sample_df["tripId"] == tripID][
        "UnixTimeMillis"
    ].to_numpy()
    pred_dfs.append(ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis))
sub_df = pd.concat(pred_dfs)

baselines = []
gts = []
for dirname in tqdm(sorted(glob.glob(f"{INPUT_PATH}/train/*/*"))):
    drive, phone = dirname.split("/")[-2:]
    tripID = f"{drive}/{phone}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv", low_memory=False)
    gt_df = pd.read_csv(f"{dirname}/ground_truth.csv", low_memory=False)
    baseline_df = ecef_to_lat_lng(tripID, gnss_df, gt_df["UnixTimeMillis"].to_numpy())
    baselines.append(baseline_df)
    gts.append(gt_df)
baselines = pd.concat(baselines)
gts = pd.concat(gts)


# In[ ]:


baselines["group"] = "train_baseline"
sub_df["group"] = "submission_baseline"
gts["group"] = "train_ground_truth"
combined = pd.concat([baselines, sub_df, gts]).reset_index(drop=True).copy()


# # This Dataset has to main areas
# - Paths from the bay area and LA. Lets split the data into the two areas

# In[ ]:


sf_paths = combined.query("LatitudeDegrees > 36").copy()
la_paths = combined.query("LatitudeDegrees < 36").copy()


# # Plot SF Paths
# - Ground Truth
# - Baseline for training paths
# - Baseline for test paths

# In[ ]:


visualize_traffic(
    sf_paths.sample(frac=0.2),
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    color_col="group",
    zoom=9,
)


# In[ ]:


visualize_traffic(
    la_paths.sample(frac=0.2),
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    color_col="group",
    zoom=9,
)


# # Post Processing Idea 1
# - Find Stopped Locations and Isolate

# In[ ]:


def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist


def add_prev_post_shift(
    df,
    lat_col="LatitudeDegrees",
    lng_col="LongitudeDegrees",
    dist_suffix="",
    sortby=["tripId", "UnixTimeMillis"],
):
    df = df.sort_values(sortby).reset_index(drop=True)
    df[f"{lat_col}_shift1"] = df.groupby(["tripId"])[lat_col].shift(1)
    df[f"{lng_col}_shift1"] = df.groupby(["tripId"])[lng_col].shift(1)
    df[f"{lat_col}_shift-1"] = df.groupby(["tripId"])[lat_col].shift(-1)
    df[f"{lng_col}_shift-1"] = df.groupby(["tripId"])[lng_col].shift(-1)

    df[f"UnixTimeMillis_shift1"] = df.groupby(["tripId"])["UnixTimeMillis"].shift(1)
    df[f"UnixTimeMillis_shift-1"] = df.groupby(["tripId"])["UnixTimeMillis"].shift(-1)

    df[f"dist_prev{dist_suffix}"] = calc_haversine(
        df[lat_col], df[lng_col], df[f"{lat_col}_shift1"], df[f"{lng_col}_shift1"]
    )
    df[f"dist_post{dist_suffix}"] = calc_haversine(
        df[lat_col], df[lng_col], df[f"{lat_col}_shift-1"], df[f"{lng_col}_shift-1"]
    )
    return df


# In[ ]:


baselines2 = add_prev_post_shift(baselines)
baselines2["UnixTimeMillis_prev_diff"] = (
    baselines2["UnixTimeMillis"] - baselines2["UnixTimeMillis_shift1"]
)
baselines2["speed_calc"] = (
    baselines2["dist_prev"] / baselines2["UnixTimeMillis_prev_diff"]
)


sub_df = add_prev_post_shift(sub_df)
sub_df["UnixTimeMillis_prev_diff"] = (
    sub_df["UnixTimeMillis"] - sub_df["UnixTimeMillis_shift1"]
)
sub_df["speed_calc"] = sub_df["dist_prev"] / sub_df["UnixTimeMillis_prev_diff"]


# # Find when the cars are not moving:

# In[ ]:


baselines2.query("dist_prev < 5")["dist_prev"].plot(
    kind="hist", bins=50, title="Distribution of Speeds < 5"
)
plt.show()


# In[ ]:


def do_postprocess(sub_df, thres=1):
    sub = sub_df.copy()
    for c, sub_stopped in sub.groupby("tripId"):
        sub_stopped = sub_stopped.loc[sub_stopped["dist_prev"] < thres].copy()
        sub_stopped["UnixTimeMillis_diff"] = sub_stopped["UnixTimeMillis"].diff()
        sub_stopped["big_timeshift"] = sub_stopped["UnixTimeMillis_diff"] > 2_000
        sub_stopped["time_group"] = sub_stopped["big_timeshift"].astype("int").cumsum()

        for stop_group, d in sub_stopped.groupby("time_group"):
            tstart, tstop = d["UnixTimeMillis"].min(), d["UnixTimeMillis"].max()
            stopped_len = len(
                sub.loc[
                    (sub["UnixTimeMillis"] >= tstart) & (sub["UnixTimeMillis"] <= tstop)
                ]
            )
            if stopped_len >= 20:
                buffer = 800
                latDegmean = sub.loc[
                    (sub["UnixTimeMillis"] >= (tstart - buffer))
                    & (sub["UnixTimeMillis"] <= (tstop + buffer))
                ]["LatitudeDegrees"].mean()
                lngDegmean = sub.loc[
                    (sub["UnixTimeMillis"] >= (tstart - buffer))
                    & (sub["UnixTimeMillis"] <= (tstop + buffer))
                ]["LongitudeDegrees"].mean()
                sub.loc[
                    (sub["UnixTimeMillis"] >= tstart)
                    & (sub["UnixTimeMillis"] <= tstop),
                    "LatitudeDegrees",
                ] = latDegmean
                sub.loc[
                    (sub["UnixTimeMillis"] >= tstart)
                    & (sub["UnixTimeMillis"] <= tstop),
                    "LongitudeDegrees",
                ] = lngDegmean
                sub.loc[
                    (sub["UnixTimeMillis"] >= tstart)
                    & (sub["UnixTimeMillis"] <= tstop),
                    "stopped",
                ] = True
    sub["stopped"] = sub["stopped"].fillna(False)
    return sub


# In[ ]:


sub = do_postprocess(sub_df, thres=1.5)


# In[ ]:


visualize_traffic(
    sub.query("LatitudeDegrees > 36").sample(frac=0.2),
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    color_col="stopped",
    zoom=10,
)


# ## Check Post Processing on Training Dataset

# In[ ]:


baselines2 = add_prev_post_shift(baselines)
baselines2["UnixTimeMillis_prev_diff"] = (
    baselines2["UnixTimeMillis"] - baselines2["UnixTimeMillis_shift1"]
)
baselines2["speed_calc"] = (
    baselines2["dist_prev"] / baselines2["UnixTimeMillis_prev_diff"]
)

baselines2_pp = do_postprocess(baselines2, thres=1.5)
scores = []
for tripID in baselines2_pp["tripId"].unique():
    score = calc_score(tripID, baselines2_pp, gts)
    scores.append(score)

mean_score = np.mean(scores)
print(f"mean_score = {mean_score:.3f}")


# # Save off results

# In[ ]:


sub.reset_index(drop=True)[ss.columns].to_csv("submission.csv", index=False)

