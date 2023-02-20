## Fetch water information
#
# starter functions for exercise 5 in notebook 1
#
# written by Jess Breda


import datajoint as dj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ratinfo = dj.create_virtual_module("intfo", "ratinfo")


def fetch_daily_water_target(animal_id, date, verbose=False):
    """
    Function for getting an animals water volume target on
    a specific date

    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04
    verbose : bool
        if you want to print restriction information

    returns
    ------
    volume_target : float
        water restriction target in mL
    """

    percent_target = fetch_daily_restriction_target(animal_id, date)
    mass = fetch_daily_mass(animal_id, date)

    volume_target = np.round((percent_target / 100) * mass, 2)

    if verbose:
        print(
            f"""On {date} {animal_id} is restricted to:
        {percent_target}% of body weight or {volume_target} mL
        """
        )

    return volume_target


def fetch_daily_restriction_target(animal_id, date):
    """
    Function for getting an animals water
    target for a specific date

    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04

    returns
    ------
    percent_target : float
        water restriction target in terms of percentage of body weight

    note
    ----
    You can also fetch this code from the registry, but it's not
    specific to the date. See code below.

    ```
    # fetch from comments section e.g. 'Mouse Water Pub 4'
    r500_registry = (ratinfo.Rats & 'ratname = "R501"').fetch1()
    comments = r500_registry['comments']

    #split after the word pub,only allow one split and
    # grab whatever follows the split
    target = float(comments.split('Pub',1)[1])
    ```
    """

    Water_keys = {"rat": animal_id, "date": date}

    # can't do fetch1 with this becaues water table
    # has a 0 entry and actual entry for every day
    # I'm taking the max to get around this
    # this needs to be address w/ DJ people
    percent_target = float((ratinfo.Water & Water_keys).fetch("percent_target").max())

    return percent_target


def fetch_daily_mass(animal_id, date):
    """
    Function for getting an animals mass on
    a specific date

    inputs
    ------
    animal_id : str,
        animal name e.g. "R501"
    date : str
        date to query in YYYY-MM-DD format, e.g. "2022-01-04

    returns
    ------
    mass : float
        weight in grams on date
    """

    Mass_keys = {"ratname": animal_id, "date": date}
    mass = float((ratinfo.Mass & Mass_keys).fetch1("mass"))

    return mass


def plot_daily_water(volume_target, rig_volume, pub_volume, animal_id, date):

    """
    Quick function for plotting water consumed in rig or pub for a day
    and marking the target with a horizontal line
    """

    df = pd.DataFrame(
        {"date": [date], "rig_volume": [rig_volume], "pub_volume": [pub_volume]}
    )

    fig, ax = plt.subplots(1, 1, figsize=(3, 4))

    # plot
    df.set_index("date").plot(kind="bar", stacked=True, color=["blue", "cyan"], ax=ax)
    ax.axhline(y=volume_target, xmin=0.2, xmax=0.8, color="black")

    # legend
    order = [1, 0]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[i] for i in order], [labels[i] for i in order], loc=(0.8, 0.75))

    # aesthetics
    _ = plt.xticks(rotation=45)
    _ = ax.set(xlabel="", ylabel="volume (mL)", title=f"{animal_id} water info")
    sns.despine()
