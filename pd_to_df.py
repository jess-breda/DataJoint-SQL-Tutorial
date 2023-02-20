### Protocol Data to Data Frame
#
# funtions used to import protocol data into python via
# DataJoint to create a trial-by-trial dataframe for
# a given animal(s)
#
# Written by Jess Breda
#


## Libraries
import datajoint as dj

dj.blob.use_32bit_dims = True
import numpy as np
import pandas as pd
from dj_utils import transform_blob


## Constants
ANIMAL_IDS = ["R500", "R501", "R502", "R503", "R600"]

MAP_SA_TO_SB = {
    12000: 3000,
    3000: 12000,
}
###############################
## === Primary Function ==== ##
###############################
def fetch_latest_training_data(animal_ids=None, drop_trial_report=False):
    """
    Function to query bdata via datajoint to get trial by trial
    protocol data for a(n) animal(s), clean it, save out and return
    as a single pandas data frame.

    inputs
    ------
    animal_ids : list, optional
        animal(s) to query database with, (default = ANIMAL_IDS)
    crashed_trials_report : bool TODO
    print_sess_id : bool TODO

    returns
    -------
    all_animals_protocol_df : data frame
        data frame containing protocol data for every session for
        every animal in animal_ids
    """
    animal_ids = ANIMAL_IDS if animal_ids is None else animal_ids
    assert type(animal_ids) == list, "animal ids must be in a list"

    animals_protocol_dfs = []

    # initate host
    bdata = dj.create_virtual_module("bdata", "bdata")

    # fetch data, clean it & convert to df for each animal
    for animal_id in animal_ids:
        subject_session_key = {"ratname": animal_id}

        # protocol data is fetched on it's own from sessions table
        # since it's n sessions x n trials/session long
        protocol_blobs = (bdata.Sessions & subject_session_key).fetch(
            "protocol_data", as_dict=True
        )

        # n session long items are fetched together
        sess_ids, dates, trials = (bdata.Sessions & subject_session_key).fetch(
            "sessid", "sessiondate", "n_done_trials"
        )

        # remove any sessions with 0 or 1 trials since they break
        # the code below
        protocol_blobs, sess_ids, dates, trials = drop_empty_sessions(
            protocol_blobs, sess_ids, dates, trials, drop_trial_report=drop_trial_report
        )

        # blob for each session -> dict for each session
        protocol_dicts = convert_to_dict(protocol_blobs)

        # asserts for length & fixes DMS specific issues in place
        pd_prepare_dicts_for_df(protocol_dicts)

        # dicts -> df & correct data types, add columns, etc.
        protocol_df = make_protocol_df(protocol_dicts, animal_id, sess_ids, dates)

        animals_protocol_dfs.append(protocol_df)

        # using dates because can have multiple sess_ids in one session
        print(
            f"fetched {len(dates)} sessions for {animal_id} with latest date {max(dates)}"
        )

    # concatenate across animals
    all_animals_protocol_df = pd.concat(animals_protocol_dfs, ignore_index=True)

    return all_animals_protocol_df


###############################
## === Supporting Functions ===
###############################
def drop_empty_sessions(pd_blobs, sess_ids, dates, trials, drop_trial_report=False):
    """
    sessions with 0 or 1 trials break the later code because
    of dimension errors, so they need to be dropped
    """

    trial_filter = (trials != 0) & (trials != 1)

    if drop_trial_report:
        print(
            f"dropping {len(pd_blobs) - np.sum(trial_filter)} sessions of {len(pd_blobs)} due to <2 trials"
        )

    pd_blobs = np.array(pd_blobs)  # list -> array needed for bool indexing
    pd_blobs = pd_blobs[trial_filter]

    sess_ids = sess_ids[trial_filter]
    dates = dates[trial_filter]
    trials = trials[trial_filter]

    return pd_blobs, sess_ids, dates, trials


def convert_to_dict(blobs):
    """
    Function that takes protocol data (pd) blob(s) from bdata
    sessions table query and converts them to python
    dictionaries using Alvaro's blob transformation code

    inputs
    ------
    blobs : list of arrays
        list returned when fetching pd data from bdata
        tables with len = n sessions

    returns
    -------
    dicts : list of dictionaries
        list of pd or peh dictionaries where len = n sessions and
        each session has a dictionary
    """
    # type of blob is indicated in nested, messy array structure
    data_type = list(blobs[0].keys())[0]
    assert data_type == "protocol_data", "unknown key pair"

    dicts = []

    for session_blob in blobs:
        sess_dict = transform_blob(session_blob[data_type])
        dicts.append(sess_dict)

    return dicts


def pd_prepare_dicts_for_df(protocol_dicts):
    """
    Function to clean up protocol data dictionary lengths,
    names & types to ensure there are no errors & interpretation
    issues upon converting it into a data frame. Has some
    DMS specific corrections.

    inputs.
    ------
    protocol_dicts : list of dicts
        list of dictionaries for one or more sessions protocol data

    modifies
    --------
    protocol_dicts : list of dicts
        corrects side vector format, updates DMS match/nonmatch
        variable names, corrects for bugs found in HistorySection.m
        that led to differing variables lengths
    """
    for isess, protocol_dict in enumerate(protocol_dicts):
        # lllrllr to [l, l, l, r....]
        protocol_dict["sides"] = list(protocol_dict["sides"])

        # if DMS, convert match/nonmatch category variable to bool
        # with more informative name
        if "dms_type" in protocol_dict:
            protocol_dict["is_match"] = protocol_dict.pop("dms_type")
            protocol_dict["is_match"] = protocol_dict["is_match"].astype(bool)

        # check to see if protocol_data is pre HistorySection.m bug fixes
        # where len(each value) was not equal. Using sa as reference length
        # template but this could lead to errors if sa has bug
        if len(protocol_dict["sa"]) != len(protocol_dict["sb"]):
            _truncate_sb_length(protocol_dict)
        if len(protocol_dict["sa"]) != len(protocol_dict["result"]):
            _fill_result_post_crash(protocol_dict)

        # test for remaining length errors (important regardless of protocol)
        lens = map(len, protocol_dict.values())
        n_unique_lens = len(set(lens))
        assert n_unique_lens == 1, "length of dict values unequal!"


def _truncate_sb_length(protocol_dict):
    """
    *** specific DMS issue ***
    Function to correct for bug in HistorySection, see commit:
    https://github.com/Brody-Lab/Protocols/commit/4a2fadb802d64b7ed66891a263a366a8d2580483
    sb vector was 1 greater than n_started_trials due to an
    appending error

    inputs
    ------
    protocol_dict : dict
        dictionary for a single session's protocol_data

    modifies
    -------
    protocol_dict : dict
        updated protocol_dict with sb column length & contents corrected
        if DMS. length only corrected if PWM2 protocol is being used
    """

    # rename for ease
    sa = protocol_dict["sa"]
    sb = protocol_dict["sb"]
    match = protocol_dict["is_match"]

    # if DMS task, can infer values
    if "is_match" in protocol_dict:
        for trial in range(len(sa)):
            if match[trial]:
                sb[trial] = sa[trial]  # update sb
            else:
                assert sa[trial] in MAP_SA_TO_SB, "sa not known"
                sb[trial] = MAP_SA_TO_SB[sa[trial]]
    else:
        print("sb values incorrect, only fixing length")

    sb = sb[0:-1]  # remove extra entry
    protocol_dict["sb"] = sb  # update


def _fill_result_post_crash(protocol_dict):
    """
    *** specific DMS issue ***
    Function to correct for bug in HistorySection, see commit:
    https://github.com/Brody-Lab/Protocols/commit/3bdde4377ffde011cc34d098acfeb77b74c9e606
    result vector was shorter than n_started_trials because program
    crashed & results_history vector was not being properly filled
    during crash clean up

    inputs
    ------
    protocol_dict : dict
        dictionary for a single session's protocol_data

    modifies
    -------
    protocol_dict : dict
        updated protocol_dict with results column length corrected to
        reflect crash trials
    """

    # rename for ease
    results = protocol_dict["result"]
    sa = protocol_dict["sa"]

    # pack with crash result value (5)
    crash_appended_results = np.ones((len(sa))) * 5
    crash_appended_results[0 : len(results)] = results

    protocol_dict["result"] = crash_appended_results


def make_protocol_df(
    protocol_dicts,
    animal_id,
    sess_ids,
    dates,
):
    """
    Converts

    inputs:
    -------
    protocol_dicts : list of dictionaries
        protocol_data dictionary for each session queried
    animal_id : str
        id of animal that protocol_data is associated with, note this
        function currently assumes 1 animal per query
    sess_ids : arr
        session ids fetched from sessions table that correspond to
        values in protocol_dicts
    dates : arr
        dates fetched from sessions table that correspond to
        values in protocol_dicts

    !note!
        pd data structure must be saved to sessions table with all
        features having same length (n_started_trials). see DMS or PWM2
        protocols' HistorySection.m for example in `make_and_send_summary`
        Early stages of testing these new protocols didn't follow this
        rule & caused bugs that are outlined & fixed via `truncate_sb_length`
        and `fill_results_post_crash`

    returns:
    -------
    sessions_protocol_df: date frame
        protocol_data for every session for an animal
    """
    session_protocol_dfs = []

    # for each session, turn protocol data dict into data frame
    # and then concatenate all sessions together
    for isess, sess_id in enumerate(sess_ids):

        protocol_df = pd.DataFrame.from_dict(protocol_dicts[isess])

        # add columns, correct data types in place
        clean_pd_df(protocol_df, animal_id, sess_id, dates[isess])

        session_protocol_dfs.append(protocol_df)

    # concat list of dfs -> one large df for all sessions
    sessions_protocol_df = pd.concat(session_protocol_dfs, ignore_index=True)

    return sessions_protocol_df


def clean_pd_df(protocol_df, animal_id, sess_id, date):
    """
    Function that takes a protocol_df for a session and cleans
    it to correct for data types and format per JRBs preferences

    inputs
    ------
    protocol_df : data frame
        protocol_data dictionary thats been converted to df for a single session
    animal_id : str
        animal id for which the session corresponds to
    sess_id : str
        id from bdata corresponding to session
    date : datetime object or str
        date corresponding to session

    modifies
    -------
    protocol_df : data frame
        (1) crashed trials removed
        (2) animal, date, session id columns added
        (3) sa/sb converted from Hz to kHz
        (4) certain columns converted to ints & categories

    TODO: integrate with n_done_trials to clip lengths to be the same for all
    other trial length variables. Often, this is 1 longer than parsed events
    history for example.
    """

    n_started_trials = len(protocol_df)

    # create trials column
    protocol_df.insert(0, "trial", np.arange(1, n_started_trials + 1, dtype=int))

    # drop any trials where dispatcher reported a crash
    protocol_df.drop(protocol_df[protocol_df["result"] == 5].index, inplace=True)

    # add animal id, data and sessid value for each trial
    protocol_df.insert(1, "animal_id", [animal_id] * len(protocol_df))
    protocol_df.insert(2, "date", [date] * len(protocol_df))
    protocol_df.insert(3, "sess_id", [sess_id] * len(protocol_df))

    # convert units to kHz
    protocol_df[["sa", "sb"]] = protocol_df[["sa", "sb"]].apply(lambda row: row / 1000)

    # create a unique pair column indicating sa,sb (eg. "12-3")
    protocol_df["sound_pair"] = protocol_df.apply(
        lambda row: str(row.sa) + ", " + str(row.sb), axis=1
    )

    # add a violations column, move it to column #6
    protocol_df["violations"] = protocol_df.apply(
        lambda row: 1 if row.result == 3 else 0, axis=1
    )
    protocol_df.insert(5, "violations", protocol_df.pop("violations"))

    # convert data types (matlab makes everything a float)
    int_columns = [
        "hits",
        "violations",
        "temperror",
        "result",
        "helper",
        "stage",
        "sess_id",
    ]
    protocol_df[int_columns] = protocol_df[int_columns].astype("Int64")

    # let pandas know if it's a category (helps with plotting)
    category_columns = ["result", "stage", "sess_id", "sound_pair"]
    protocol_df[category_columns] = protocol_df[category_columns].astype("category")
