import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


#####################
###     PLOTS     ###
#####################


def plot_trials(df, ax, title="", legend=False, xaxis_label=True):
    """
    Plot the number of trials completed per day and the trial rate

    params
    ------
    df : pd.DataFrame
        dataframe with columns `date`, `n_done_trials`, `trial_rate` with
        dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = False)
        whether to include the legend or not
    """
    trial_melt = df.melt(
        id_vars=["date"],
        value_name="trial_var",
        value_vars=["n_done_trials", "trial_rate"],
    )
    sns.lineplot(
        data=trial_melt,
        x="date",
        y="trial_var",
        hue="variable",
        marker="o",
        ax=ax,
    )

    # aethetics
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    _ = ax.set(ylabel="Count || Per Hr", xlabel="", title=title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.5)

    return None


def plot_mass(df, ax, title="", xaxis_label=True):
    """
    Plot the mass of the animal over date range in df

    params
    ------
    df : pd.DataFrame
        daily dataframe with columns `date`, `mass` with dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    """

    sns.lineplot(data=df, x="date", y="mass", marker="o", color="k", ax=ax)

    # aethetics
    set_date_x_ticks(ax, xaxis_label)
    ax.grid(alpha=0.5)
    ax.set(ylabel="Mass [g]", xlabel="", title=title)

    return None


def plot_water_restriction(df, ax, title="", legend=True, xaxis_label=True):
    """
    Plot the rig, pub and restriction target volume over data
    range in df

    params
    ------
    df : pd.DataFrame
        dataframe with columns `date`, `rig_volume`, `pub_volume`
        and `volume_target` with dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = True)
        whether to include the legend or not
    """

    # stacked bar chart only works with df.plot (not seaborn)
    columns_to_plot = ["date", "rig_volume", "pub_volume"]
    df[columns_to_plot].plot(
        x="date",
        kind="bar",
        stacked=True,
        color=["blue", "cyan"],
        ax=ax,
    )

    # iterate over dates to plot volume target black line
    for i, row in df.reset_index().iterrows():
        ax.hlines(y=row["volume_target"], xmin=i - 0.35, xmax=i + 0.35, color="black")
    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    ax.set(title=title, xlabel="", ylabel="Volume [mL]")

    return None


def plot_rig_tech(df, ax, title="", legend=False, xaxis_label=True):
    """
    Plot the tech and rig id over data range in df

    params
    ------
    df : pd.DataFrame
        dataframe with columns `date`, `rigid` and `tech` with
        dates as row index
    ax : matplotlib.axes.Axes
        axes to plot on
    title : str (optional, default = "")
        title for the plot
    legend : bool (optional, default = False)
        whether to include the legend or not
    """
    sns.lineplot(data=df, x="date", y="rigid", marker="o", color="gray", ax=ax)
    sns.lineplot(data=df, x="date", y="tech", marker="o", color="purple", ax=ax)

    set_date_x_ticks(ax, xaxis_label)
    _ = ax.set(ylabel="Tech || Rig", xlabel="", title=title)
    ax.grid()

    return None


def plot_performance(df, ax, title="", legend=True, xaxis_label=True):
    """ """
    sns.lineplot(
        data=df,
        x="date",
        y="hit_rate",
        marker="o",
        color="darkgreen",
        label="hit",
        ax=ax,
    )
    sns.lineplot(
        data=df,
        x="date",
        y="viol_rate",
        marker="o",
        color="orangered",
        label="viol",
        ax=ax,
    )

    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    set_legend(ax, legend)
    ax.grid(alpha=0.5)
    ax.set(ylim=(0, 1), ylabel="Performance", xlabel="", title=title)

    return None


def plot_side_bias(df, ax, title="", xaxis_label=True):
    """ """
    sns.lineplot(
        data=df,
        x="date",
        y="side_bias",
        color="lightseagreen",
        marker="o",
        ax=ax,
    )
    ax.axhline(0, color="k", linestyle="--", zorder=1)

    # aesthetics
    set_date_x_ticks(ax, xaxis_label)
    ax.grid(alpha=0.5)
    ax.set(ylim=(-1, 1), ylabel="< - Left | Right ->", xlabel="", title=title)

    return None


#####################
###  PLOT HELPERS ###
#####################


def return_date_window(latest_date=None, n_days_back=None):
    """
    Function to create a date window for querying the DataJoint
    SQL database. Pairs nicely with `fetch_latest_trials_data`
    or `fetch_daily_summary_data`

    params
    ------
    latest_date : str  (optional, default = None)
        latest date to include in the window, defaults to today
        if left empty
    n_days_back : int (optional, default = None)
        number of days back from `latest_date` to include,
        defaults to all days if left empty

    Note: if you are out of range of your table (e.g min date)
    is before the start of training) it's okay.

    returns
    ------
    min_date : str
        minimum date in the specified date window
    max_date : str
        maximum date in the specified date window

    example usage
    ------------
    `return_date_window(latest_date='2022-11-3', n_days_back=8)`
    `return_date_window(latest_date=None, n_days_back=7)`
    """

    if latest_date:
        date_max = pd.to_datetime(latest_date)
    else:
        date_max = pd.Timestamp.today()

    if n_days_back:
        date_min = date_max - pd.Timedelta(days=n_days_back)
        date_min = date_min.strftime("%Y-%m-%d")
    else:
        date_min = n_days_back  # none

    return date_min, date_max.strftime("%Y-%m-%d")


def make_fig(figsize=(10, 3)):
    "Quick fx for subplot w/ 10 x 3 size default"

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def set_date_x_ticks(ax, xaxis_label):
    "Quick fx for rotating xticks on date axis using ax object (not plt.)"

    if xaxis_label:
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels(ax.get_xticklabels(), ha="right", rotation=45)
    else:  # turn off the labels
        ax.set_xticklabels([])


def set_legend(ax, legend):
    "Quick fx for setting legend on/off using ax object (not plt.)"
    if legend:
        ax.legend(frameon=False, borderaxespad=0)
    else:
        ax.get_legend().remove()
