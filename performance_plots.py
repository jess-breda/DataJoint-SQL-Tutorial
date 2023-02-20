## Performance plotting utils
#
# functions used to plot (or prep for plotting)
# animal training progress from protocol data dataframe
#
# written by Jess Breda
#

## Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##############################
## ===Support Functions === ##
##############################


def filter_for_date_window(df, latest_date=None, n_days_back=None):
    """
    Function to filter a data frame for a given date window and
    return a date frame in that window. If no parameters are
    specified, will return the input data frame

    inputs
    ------
    df : data frame
        data frame to be filtered with a `date` column of
        of type pd_datetime
    latest_date : str or pd_datetime (optional, default = None)
        latest date to include in the window, defaults to
        the latest date in the data frame
    n_days_back : int (optional, default = None)
        number of days back from `latest_date` to include,
        defaults to all days

    returns
    -------
    df : data frame
        date frame that only includes dates starting at the
        `latest_date` until `n_days_back`

    example usage
    ------------
    `filter_for_date_window(df, latest_date='2022-11-3', n_days_back=8)`
    `filter_for_date_window(df, latest_date=None, n_days_back=7)

    """

    # grab everything prior to specified date.
    # if not specified, use the latest date
    if latest_date:
        latest_date = pd.to_datetime(latest_date)
    else:
        latest_date = df.date.max()
    df = df[(df.date <= latest_date)]

    # starting from latest_date, grab n_days_back
    if n_days_back:
        earliest_date = latest_date - pd.Timedelta(days=n_days_back)
        df = df[(df.date > earliest_date)]

    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def calculate_bias_history(df, latest_date=None, n_days_back=None):
    # select date range
    df = filter_for_date_window(df, latest_date=latest_date, n_days_back=n_days_back)

    # calculate bias
    bias_df = pd.DataFrame(columns=["date", "bias"])

    for date, date_df in df.groupby("date"):

        side_perf = date_df.groupby("sides").mean().hits.reset_index()

        # always make sure left row is first
        side_perf = side_perf.sort_values(by="sides")

        # left hits - right hits, 2 decimal points
        bias = round(side_perf.hits.iloc[0] - side_perf.hits.iloc[1], 2)

        bias_df = pd.concat(
            [bias_df, pd.DataFrame({"date": [date], "bias": [bias]})], ignore_index=True
        )

    return bias_df


def create_palette_given_sounds(df):
    """
    Function to allow for assignment of specific colors to a sound pair
    that is consistent across sessions where number of unique pairs varies
    """
    palette = []
    sound_pairs = df.sound_pair.unique()

    sound_pair_colormap = {
        "3.0, 3.0": "skyblue",
        "12.0, 12.0": "steelblue",
        "3.0, 12.0": "thistle",
        "12.0, 3.0": "mediumorchid",
    }

    for sp in sound_pairs:
        palette.append(sound_pair_colormap[sp])
    return palette


############################
## === Plot Functions === ##
############################


def plot_trials(df, ax, title=None, **kwargs):
    title = "Trial Plot" if title is None else title
    sns.lineplot(data=df.groupby("date").max().trial, ax=ax, **kwargs)

    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="trials per session", title=title)
    sns.despine()


def plot_hits(df, ax, title=None, **kwargs):
    title = "Hit Plot" if title is None else title
    sns.lineplot(data=df, x="date", y="hits", errorbar=None, ax=ax, **kwargs)

    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="fraction correct", title=title, ylim=[0, 1])
    sns.despine()


def plot_hits_and_viols(df, ax, title=None):
    title = "Hit & Viol Plot" if title is None else title

    sns.lineplot(data=df, x="date", y="hits", color="seagreen", errorbar=None, ax=ax)
    sns.lineplot(
        data=df, x="date", y="violations", color="firebrick", errorbar=None, ax=ax
    )

    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="fraction correct | viol", title=title, ylim=[0, 1])
    ax.legend(
        ["hits", "viols"],
        bbox_to_anchor=(1, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )
    sns.despine()


def plot_bias_history(df, ax, latest_date=None, n_days_back=None, **kwargs):

    bias_df = calculate_bias_history(
        df, n_days_back=n_days_back, latest_date=latest_date
    )

    # if having issues with time plots, try this
    # bias_df["date"] = bias_df["date"].astype(str)

    sns.lineplot(
        data=bias_df,
        x="date",
        y="bias",
        errorbar=None,
        marker="o",
        markersize=7,
        ax=ax,
        **kwargs,
    )
    ax.axhline(0, color="k", linestyle="--", zorder=1)

    _ = plt.xticks(rotation=45)
    _ = ax.set(ylabel="<-- right bias | left bias -->", title="Side Bias", ylim=[-1, 1])
    sns.despine()
    return bias_df


def plot_stim_in_use(df, ax):
    """
    Function to plot which sa,sb pairs are currently
    active for an animal.
    """

    df = df[df.date == df.date.max()]

    stim_pairs_str = df.sound_pair.unique()
    stim_pairs = []

    # iterate over list of strings and reformat into floats
    # for plotting
    for sp in stim_pairs_str:
        sa, sb = sp.split(", ")  # ['sa, sb'] -> 'sa', 'sb'
        stim_pairs.append((float(sa), float(sb)))

    # assigns specific colors to sound pairs to keep theme
    color_palette = create_palette_given_sounds(df)

    for s, c in zip(stim_pairs, color_palette):
        ax.scatter(s[0], s[1], marker=",", s=300, c=c, alpha=0.75)

    # Match/non-match boundary line
    plt.axline((0, 1), slope=1, color="lightgray", linestyle="--")
    plt.axline((1, 0), slope=1, color="lightgray", linestyle="--")

    # plot range & aesthetics
    sp_min, sp_max = np.min(stim_pairs), np.max(stim_pairs)
    stim_range = [sp_min, sp_max]
    x_lim = [sp_min - 3, sp_max + 3]
    y_lim = [sp_min - 3, sp_max + 3]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xticks(stim_range)
    ax.set_yticks(stim_range)
    ax.set(title="Stimulus Pairs", xlabel="Sa [kHz]", ylabel="Sb [kHz]")
    sns.despine()


def plot_pair_performance(df, ax, title=None):
    """
    Function for plotting hit rate over time for each sa,sb pair
    """

    title = "Pair Perf Plot" if title is None else title

    perf_by_sound = df.pivot_table(
        index="date", columns="sound_pair", values="hits", aggfunc="mean"
    )

    colors = {
        "3.0, 3.0": "skyblue",
        "12.0, 12.0": "steelblue",
        "3.0, 12.0": "thistle",
        "12.0, 3.0": "mediumorchid",
    }

    # using df.plot since it doesn't fill in days without stim like
    # seaborn does
    perf_by_sound.plot.line(color=colors, ax=ax, rot=45, style=".-")

    # 50 and 75% lines
    ax.axhline(0.5, color="k", linestyle="--", zorder=1)
    ax.axhline(0.75, color="gray", linestyle="--", zorder=1)

    _ = ax.set(ylim=[0, 1], ylabel="fraction correct", title=title)
    sns.despine()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0, frameon=False)
