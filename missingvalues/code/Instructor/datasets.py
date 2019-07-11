"""
Author      : Huey Fields
Class       : HMC CS 181R
Date        : 2019 June 7
Description : ML Datasets
"""

# python modules
import os

# numpy module
import numpy as np

# pandas module
import pandas as pd

# scikit-learn module
from sklearn.preprocessing import LabelEncoder

READ_FOLDER = os.path.join("..", "..", "data", "processed")

######################################################################
# functions
######################################################################

def credit(multiclass = False):
    """Load sqf dataset"""

    target_column = "TARGET"

    target_names = ["NO DEFAULT", "DEFAULT"]
    labels = [0, 1]

    # Training data
    df = pd.read_csv(os.path.join(READ_FOLDER, 'credit.csv'))

    # Drop columns with too many nans
    #drop_columns = df.isnull().sum()[df.isnull().sum() > 5000].index

    #df = df.drop(drop_columns, axis=1)

    # -------------------------- ENCODING -------------------------- #

    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                # Train on the training data
                le.fit(df[col])
                # Transform both training and testing data
                df[col] = le.transform(df[col])
                
                # Keep track of how many columns were label encoded
                le_count += 1
                
    print('%d columns were label encoded.' % le_count)

    # one-hot encoding of categorical variables
    df = pd.get_dummies(df)

    # -------------------------- ENCODING -------------------------- #


    # -------------------------- ANOMALIES -------------------------- #

    # replace anomalies
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

    # -------------------------- ANOMALIES -------------------------- #


    # -------------------------- ALIGNMENT -------------------------- #

    # -------------------------- ALIGNMENT -------------------------- #

    #df = df.dropna(subset=[target_column])
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # get features, labels, and feature_names
    feature_names = X.columns

    for column in X:
        X[column] = X[column].astype(float)

    return X, y, labels, target_names, feature_names

def earlysqf(multiclass = False):
    """Load pre-2015 sqf dataset"""

    target_column = "frisked"

    target_names = ["Frisked", "No frisk"]
    labels = [1, -1]

    target_column_multiclass = "race"

    target_names_multiclass = [
        "Black",
        # "White Hispanic",
        "White", 
        # "Black Hispanic", 
        # "Asian / Pacific Islander",
        "Other",
        # "Unknown",
        # "American Indian/Alaskan Native",
    ]

    labels_multiclass = [
        "BLACK",
        # "WHITE HISPANIC",
        "WHITE", 
        # "BLACK HISPANIC", 
        # "ASIAN / PACIFIC ISLANDER",
        "OTHER",
        # "UNKNOWN",
        # "AMERICAN INDIAN/ALASKAN NATIVE",
    ]

    # read csv
    df = pd.read_csv(os.path.join(READ_FOLDER, "2010.csv"))

    # Filter to only march entries
    #df = df.replace(to_replace = np.nan, value = '(null)')
    #df = df[df["datestop"].astype(str).str[0] == "3"]

    # drop uninformative columns
    drop_columns = [
        "year",
        "ser_num",
        "datestop",
        "timestop",
        "recstat",
        "crimsusp",
        "explnstp",
        "othpers",
        "arstmade",
        "arstoffn",
        "sumissue",
        "sumoffen",
        "compyear",
        "comppct",
        "officrid",
        "searched",
        "contrabn",
        "adtlrept",
        "pistol",
        "riflshot",
        "asltweap",
        "knifcuti",
        "machgun",
        "othrweap",
        "pf_hands",
        "pf_wall",
        "pf_grnd",
        "pf_drwep",
        "pf_ptwep",
        "pf_baton",
        "pf_hcuff",
        "pf_pepsp",
        "pf_other",
        "radio",
        "rf_vcrim",
        "rf_othsw",
        "rf_attir",
        "rf_vcact",
        "rf_rfcmp",
        "rf_verbl",
        "rf_knowl",
        "sb_hdobj",
        "sb_outln",
        "sb_admis",
        "sb_other",
        "repcmd",
        "revcmd",
        "rf_furt",
        "rf_bulg",
        "offverb",
        "offshld",
        "dob",
        "othfeatr",
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
        "xcoord",
        "ycoord",
        "dettypcm",
        "linecm",
    ]

    # Map race to BLACK, WHITE, and OTHER
    race_map = {
        "BLACK": "BLACK",
        "WHITE": "WHITE",
        "BLACK-HISPANIC": "OTHER",
        "WHITE-HISPANIC": "OTHER",
        "OTHER": "OTHER",
        "ASIAN/PACIFIC ISLANDER": "OTHER",
        "UNKNOWN": "OTHER",
        "AMERICAN INDIAN/ALASKAN NATIVE": "OTHER",
    }

    df["race"] = df["race"].map(race_map)

    # Combine height columns into single measurement in inches
    height = 12 * df["ht_feet"] + df["ht_inch"]
    height.name = "height"
    df = df.drop(["ht_feet"], axis = 1)
    df = df.drop(["ht_inch"], axis = 1)
    df = df.join(height)

    # Drop single-class target column if predicting multiclass
    if multiclass:
        drop_columns += [target_column]
    
    df = df.drop(drop_columns, axis=1)

    # Replace blank columns with NaN
    df = df.replace(to_replace = [' ', '(null)'], value = np.nan)

    # map inout to integers
    inout_map = {"O": 0, "I": 1}
    df["inout"] = df["inout"].map(inout_map)

    # one-hot encode various columns
    one_hot_columns = [
        "trhsloc",
        "typeofid",
        "race",
        "haircolr",
        "eyecolor",
        "build",
        "city",
        "detailcm",
    ]

    # If running multiclass do not one-hot encode the multiclass target column
    if multiclass and target_column_multiclass in one_hot_columns:
        one_hot_columns.remove(target_column_multiclass)

    for column in one_hot_columns:
        one_hot = pd.get_dummies(df[column], prefix=column)
        df = df.drop(column, axis=1)

        df = df.join(one_hot)

    # map target column to +/-1
    target_map = {"Y": 1, "N": -1}

    if not multiclass:
        df[target_column] = df[target_column].map(target_map)

    # map Y/N columns to integers
    YN_map = {"Y": 1, "N": 0}
    
    YN_columns = [
        "offunif",
        "ac_rept",
        "ac_inves",
        "ac_proxm",
        "cs_objcs",
        "cs_descr",
        "cs_casng",
        "cs_lkout",
        "cs_cloth",
        "cs_drgtr",
        "ac_evasv",
        "ac_assoc",
        "cs_furtv",
        "ac_cgdir",
        "cs_vcrim",
        "cs_bulge",
        "cs_other",
        "ac_incid",
        "ac_time",
        "ac_stsnd",
        "ac_other",
    ]

    for column in YN_columns:
        df[column] = df[column].map(YN_map)

    # map sex to integers
    sex_map = {"F": 0, "M": 1}
    df["sex"] = df["sex"].map(sex_map)

    # Drop any samples with nan values
    df = df.dropna()

    # drop samples with missing labels
    if multiclass:
        #df = df.dropna(subset=[target_column_multiclass])
        X = df.drop(target_column_multiclass, axis=1)
        y = df[target_column_multiclass]

    else:
        #df = df.dropna(subset=[target_column])
        X = df.drop(target_column, axis=1)
        y = df[target_column]

    # get features, labels, and feature_names
    feature_names = X.columns

    for column in X:
        X[column] = X[column].astype(float)

    # Return multiclass labels and targets if in multiclass
    if multiclass:
        labels = labels_multiclass
        target_names = target_names_multiclass

    return X, y, labels, target_names, feature_names

def earlysqf_simplified(multiclass = False):
    """Load pre-2015 sqf dataset"""

    target_column = "frisked"

    target_names = ["Frisked", "No frisk"]
    labels = [1, -1]

    target_column_multiclass = "race"

    target_names_multiclass = [
        "Black",
        # "White Hispanic",
        "White", 
        # "Black Hispanic", 
        # "Asian / Pacific Islander",
        "Other",
        # "Unknown",
        # "American Indian/Alaskan Native",
    ]

    labels_multiclass = [
        "BLACK",
        # "WHITE HISPANIC",
        "WHITE", 
        # "BLACK HISPANIC", 
        # "ASIAN / PACIFIC ISLANDER",
        "OTHER",
        # "UNKNOWN",
        # "AMERICAN INDIAN/ALASKAN NATIVE",
    ]

    # read csv
    df = pd.read_csv(os.path.join(READ_FOLDER, "2010_sqf_m35.csv"))

    # Filter to only march entries
    df = df.replace(to_replace = np.nan, value = '(null)')
    df = df[df["datestop"].astype(str).str[0] == "3"]

    # drop uninformative columns
    drop_columns = [
        "datestop",
        "timestop",
        "arstmade",
        "sumissue",
        "searched",
        "contrabn",
        "radio",
        "pf",
        "weap",
    ]

    # Drop single-class target column if predicting multiclass
    if multiclass:
        drop_columns += [target_column]
    
    df = df.drop(drop_columns, axis=1)

    # Replace blank columns with NaN
    df = df.replace(to_replace = [' ', '(null)'], value = np.nan)

    # Map race to BLACK, WHITE, and OTHER
    race_map = {
        "BLACK": "BLACK",
        "WHITE": "WHITE",
        "BLACK-HISPANIC": "OTHER",
        "WHITE-HISPANIC": "OTHER",
        "OTHER": "OTHER",
        "ASIAN/PACIFIC ISLANDER": "OTHER",
        "UNKNOWN": "OTHER",
        "AMERICAN INDIAN/ALASKAN NATIVE": "OTHER",
    }

    df["race"] = df["race"].map(race_map)

    # one-hot encode various columns
    one_hot_columns = [
        "race",
        "haircolr",
        "eyecolor",
        "build",
        "city",
        "location",
        "typeofid",
    ]

    # If running multiclass do not one-hot encode the multiclass target column
    if multiclass and target_column_multiclass in one_hot_columns:
        one_hot_columns.remove(target_column_multiclass)

    for column in one_hot_columns:
        one_hot = pd.get_dummies(df[column], prefix=column)
        df = df.drop(column, axis=1)

        df = df.join(one_hot)

    # map target column to +/-1
    target_map = {1: 1, 0: -1}

    if not multiclass:
        df[target_column] = df[target_column].map(target_map)

    # map sex to integers
    sex_map = {"F": 0, "M": 1}
    df["sex"] = df["sex"].map(sex_map)

    # drop samples with missing labels
    if multiclass:
        df = df.dropna(subset=[target_column_multiclass])
        X = df.drop(target_column_multiclass, axis=1)
        y = df[target_column_multiclass]

    else:
        df = df.dropna(subset=[target_column])
        X = df.drop(target_column, axis=1)
        y = df[target_column]

    # get features, labels, and feature_names
    feature_names = X.columns

    for column in X:
        X[column] = X[column].astype(float)

    # Return multiclass labels and targets if in multiclass
    if multiclass:
        labels = labels_multiclass
        target_names = target_names_multiclass

    return X, y, labels, target_names, feature_names

def sqf(multiclass = False):
    """Load sqf dataset"""

    target_column = "FRISKED_FLAG"

    target_names = ["No frisk", "Frisked"]
    labels = [-1, 1]

    target_column_multiclass = "SUSPECT_RACE_DESCRIPTION"

    target_names_multiclass = [
        "White", 
        "Black Hispanic", 
        "White Hispanic",
        "American Indian/Alaskan Native",
        "Black",
        "Asian/Pacific Islander",
    ]

    labels_multiclass = [
        "WHITE", 
        "BLACK HISPANIC", 
        "WHITE HISPANIC",
        "AMERICAN INDIAN/ALAKAN NATIVE",
        "BLACK",
        "ASIAN / PACIFIC ISLANDER",
    ]

    # read csv
    df = pd.read_csv(os.path.join(READ_FOLDER, "sqf-2018.csv"))
    df2 = pd.read_csv(os.path.join(READ_FOLDER, "sqf-2017.csv"))
    df = df.append(df2, ignore_index = True)

    # drop uninformative columns
    drop_columns = [
        "STOP_FRISK_ID", 
        "STOP_FRISK_DATE", 
        "LOCATION_IN_OUT_CODE",
        "OFFICER_NOT_EXPLAINED_STOP_DESCRIPTION",
        "SUSPECT_ARREST_OFFENSE",
        "SUMMONS_ISSUED_FLAG",
        "SUMMONS_OFFENSE_DESCRIPTION",
        "ID_CARD_IDENTIFIES_OFFICER_FLAG",
        "SHIELD_IDENTIFIES_OFFICER_FLAG",
        "VERBAL_IDENTIFIES_OFFICER_FLAG",
        "WEAPON_FOUND_FLAG",
        "SEARCHED_FLAG",
        "OTHER_CONTRABAND_FLAG",
        "FIREARM_FLAG",
        "KNIFE_CUTTER_FLAG",
        "OTHER_WEAPON_FLAG",
        "PHYSICAL_FORCE_CEW_FLAG",
        "PHYSICAL_FORCE_DRAW_POINT_FIREARM_FLAG",
        "PHYSICAL_FORCE_HANDCUFF_SUSPECT_FLAG",
        "PHYSICAL_FORCE_OC_SPRAY_USED_FLAG",
        "PHYSICAL_FORCE_OTHER_FLAG",
        "PHYSICAL_FORCE_RESTRAINT_USED_FLAG",
        "PHYSICAL_FORCE_VERBAL_INSTRUCTION_FLAG",
        "PHYSICAL_FORCE_WEAPON_IMPACT_FLAG",
        "BACKROUND_CIRCUMSTANCES_VIOLENT_CRIME_FLAG",
        "BACKROUND_CIRCUMSTANCES_SUSPECT_KNOWN_TO_CARRY_WEAPON_FLAG",
        "SUSPECTS_ACTIONS_CASING_FLAG",
        "SUSPECTS_ACTIONS_CONCEALED_POSSESSION_WEAPON_FLAG",
        "SUSPECTS_ACTIONS_DECRIPTION_FLAG",
        "SUSPECTS_ACTIONS_DRUG_TRANSACTIONS_FLAG",
        "SUSPECTS_ACTIONS_IDENTIFY_CRIME_PATTERN_FLAG",
        "SUSPECTS_ACTIONS_LOOKOUT_FLAG",
        "SUSPECTS_ACTIONS_OTHER_FLAG",
        "SUSPECTS_ACTIONS_PROXIMITY_TO_SCENE_FLAG",
        "SEARCH_BASIS_ADMISSION_FLAG",
        "SEARCH_BASIS_CONSENT_FLAG",
        "SEARCH_BASIS_HARD_OBJECT_FLAG",
        "SEARCH_BASIS_INCIDENTAL_TO_ARREST_FLAG",
        "SEARCH_BASIS_OTHER_FLAG",
        "SEARCH_BASIS_OUTLINE_FLAG",
        "DEMEANOR_CODE",
        "DEMEANOR_OF_PERSON_STOPPED",
        "SUSPECT_OTHER_DESCRIPTION",
        "STOP_LOCATION_PRECINCT",
        "STOP_LOCATION_SECTOR_CODE",
        "STOP_LOCATION_APARTMENT",
        "STOP_LOCATION_FULL_ADDRESS",
        "STOP_LOCATION_PREMISES_NAME",
        "STOP_LOCATION_STREET_NAME",
        "STOP_LOCATION_X",
        "STOP_LOCATION_Y",
        
        "SUSPECT_ARRESTED_FLAG",
        
        "STOP_LOCATION_ZIP_CODE",
        
        #"SUSPECT_REPORTED_AGE",
        #"SUSPECT_HEIGHT",
        #"STOP_DURATION_MINUTES",
        #"OBSERVED_DURATION_MINUTES",
        #"SUSPECT_WEIGHT",
        #"SUSPECT_SEX",
        
    ]

    # Drop single-class target column if predicting multiclass
    if multiclass:
        drop_columns += [target_column]
    
    df = df.drop(drop_columns, axis=1)

    # Replace "(null)" with NaN
    df = df.replace(to_replace = ['(null)', ' '], value = np.nan)

    # Drop rows with no time and write time as hour only
    df = df.dropna(subset = ["STOP_FRISK_TIME"])
    df["STOP_FRISK_TIME"] = df["STOP_FRISK_TIME"].apply(lambda x: x[0:x.index(":")])

    # one-hot encode various columns
    one_hot_columns = [
        "ISSUING_OFFICER_RANK",
        "SUPERVISING_OFFICER_RANK",
        "RECORD_STATUS_CODE",
        "ISSUING_OFFICER_COMMAND_CODE",
        "SUPERVISING_OFFICER_COMMAND_CODE",
        "STOP_WAS_INITIATED",
        "JURISDICTION_CODE",
        "JURISDICTION_DESCRIPTION",
        "SUSPECTED_CRIME_DESCRIPTION",
        "SUSPECT_RACE_DESCRIPTION",
        "SUSPECT_BODY_BUILD_TYPE",
        "SUSPECT_EYE_COLOR",
        "SUSPECT_HAIR_COLOR",
        "STOP_LOCATION_PATROL_BORO_NAME",
        "STOP_LOCATION_BORO_NAME",
        "MONTH2", 
        "DAY2",
        "YEAR2",
        "STOP_FRISK_TIME",
        
        
    ]

    # If running multiclass do not one-hot encode the multiclass target column
    if multiclass and target_column_multiclass in one_hot_columns:
        one_hot_columns.remove(target_column_multiclass)

    for column in one_hot_columns:
        one_hot = pd.get_dummies(df[column], prefix=column)
        df = df.drop(column, axis=1)

        df = df.join(one_hot)

    # map Y/N columns to integers
    YN_map = {"Y": 1, "N": 0}
    
    YN_columns = [
        "SUPERVISING_ACTION_CORRESPONDING_ACTIVITY_LOG_ENTRY_REVIEWED",
        "OFFICER_EXPLAINED_STOP_FLAG",
        "OTHER_PERSON_STOPPED_FLAG",
        "OFFICER_IN_UNIFORM_FLAG",
    ]

    for column in YN_columns:
        df[column] = df[column].map(YN_map)

    # map target column to +/-1
    target_map = {"Y": 1, "N": -1}

    if not multiclass:
        df[target_column] = df[target_column].map(target_map)

    # map sex to integers
    sex_map = {"FEMALE": 0, "MALE": 1}
    df["SUSPECT_SEX"] = df["SUSPECT_SEX"].map(sex_map)

    # drop samples with missing labels
    if multiclass:
        df = df.dropna(subset=[target_column_multiclass])
        X = df.drop(target_column_multiclass, axis=1)
        y = df[target_column_multiclass]

    else:
        df = df.dropna(subset=[target_column])
        X = df.drop(target_column, axis=1)
        y = df[target_column]

    # get features, labels, and feature_names
    feature_names = X.columns

    for column in X:
        X[column] = X[column].astype(float)

    # Return multiclass labels and targets if in multiclass
    if multiclass:
        labels = labels_multiclass
        target_names = target_names_multiclass

    return X, y, labels, target_names, feature_names

def ca_statepatrol():
    """Load CA State Patrol Open Policing dataset"""

    target_column = "search_conducted"

    target_names = ["No search", "Search"]
    labels = [-1, 1]

    # read csv
    df = pd.read_csv(os.path.join(READ_FOLDER, "ca_statewide_2019_02_25.csv"))

    # drop uninformative columns
    drop_columns = [
        "raw_row_number",
        "date",
        "district",
        "violation",
        "arrest_made",
        "citation_issued",
        "warning_issued",
        "outcome",
        "contraband_found",
        "frisk_performed",
        "search_person",
        "search_basis",
        "reason_for_search",
    ]
    
    df = df.drop(drop_columns, axis=1)

    # Replace "NA" with NaN
    df = df.replace(to_replace = ['NA', ' '], value = np.nan)

    # one-hot encode various columns
    one_hot_columns = [
        "county_name",
        "subject_race",
        "department_name",
        "type",
        "reason_for_stop",
    ]

    for column in one_hot_columns:
        one_hot = pd.get_dummies(df[column], prefix=column)
        df = df.drop(column, axis=1)

        df = df.join(one_hot)

    # map True/False columns to integers
    TF_map = {True: 1, False: -1}
    
    TF_columns = [
        "search_conducted"
    ]

    for column in TF_columns:
        df[column] = df[column].map(TF_map)

    # map sex to integers
    sex_map = {"female": 0, "male": 1}
    df["subject_sex"] = df["subject_sex"].map(sex_map)

    # drop samples with missing labels
    df = df.dropna(subset=[target_column])

    # get features, labels, and feature_names
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    feature_names = X.columns

    for column in df:
        df[column] = df[column].astype(float)

    return X, y, labels, target_names, feature_names

def ca_sandiego():
    """Load CA San Diego Open Policing dataset"""

    target_column = "search_conducted"

    target_names = ["No search", "Search conducted"]
    labels = [-1, 1]

    # read csv
    df = pd.read_csv(os.path.join(READ_FOLDER, "ca_san_diego_2019_02_25.csv"))

    # Filter to data from a certain time period (replacing NaNs with "(null)" to avoid string method errors)
    df = df.replace(to_replace = np.nan, value = '(null)')
    df = df[df["date"].str.contains("2017")]
    #df = df[df["date"].str[0] == "3"]

    # drop uninformative columns
    drop_columns = [
        "raw_row_number",
        "date",
        "outcome",
        "reason_for_search",
        "search_vehicle",
        "search_person",
        "search_basis",
        "type",
    ]
    
    df = df.drop(drop_columns, axis=1)

    # Replace missing values with NaN
    df = df.replace(to_replace = ['NA', '(null)', ' '], value = np.nan)

    # Drop rows with no time and write time as hour only
    df = df.dropna(subset = ["time"])
    df["time"] = df["time"].apply(lambda x: x[0:x.index(":")])

    # one-hot encode various columns
    one_hot_columns = [
        "subject_race",
        "reason_for_stop",
        "service_area",
        "time",
        
    ]

    for column in one_hot_columns:
        one_hot = pd.get_dummies(df[column], prefix=column)
        df = df.drop(column, axis=1)

        df = df.join(one_hot)

    # map True/False columns to integers
    TF_map = {True: 1, False: -1}
    
    TF_columns = [
        target_column,
        "citation_issued",
        "arrest_made",
        "contraband_found",
        "warning_issued",
    ]

    for column in TF_columns:
        df[column] = df[column].map(TF_map)

    # map sex to integers
    sex_map = {"female": -1, "male": 1}
    df["subject_sex"] = df["subject_sex"].map(sex_map)

    # drop samples with missing labels
    df = df.dropna(subset=[target_column])

    # get features, labels, and feature_names
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    feature_names = X.columns

    for column in df:
        df[column] = df[column].astype(float)

    return X, y, labels, target_names, feature_names

def ca_sanfrancisco():
    """Load CA San Francisco Open Policing dataset"""

    target_column = "search_conducted"

    target_names = ["No search", "Search conducted"]
    labels = [-1, 1]

    # read csv
    df = pd.read_csv(os.path.join(READ_FOLDER, "ca_san_francisco_2019_02_25.csv"))

    # drop uninformative columns
    drop_columns = [
        "raw_row_number",
        "date",
        "time",
        "location",
        "lat",
        "lng",
        "district",
        "arrest_made",
        "citation_issued",
        "warning_issued",
        "outcome",
        "contraband_found",
        "search_vehicle",
        "search_basis",
        "type",
    ]
    
    df = df.drop(drop_columns, axis=1)

    # Replace "NA" with NaN
    df = df.replace(to_replace = ['NA', ' '], value = np.nan)

    # one-hot encode various columns
    one_hot_columns = [
        "subject_race",
        "reason_for_stop",
    ]

    for column in one_hot_columns:
        one_hot = pd.get_dummies(df[column], prefix=column)
        df = df.drop(column, axis=1)

        df = df.join(one_hot)

    # map True/False columns to integers
    TF_map = {True: 1, False: -1}
    
    TF_columns = [
        target_column,
    ]

    for column in TF_columns:
        df[column] = df[column].map(TF_map)

    # map sex to integers
    sex_map = {"female": -1, "male": 1}
    df["subject_sex"] = df["subject_sex"].map(sex_map)

    # drop samples with missing labels
    df = df.dropna(subset=[target_column])

    # get features, labels, and feature_names
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    feature_names = X.columns

    for column in df:
        df[column] = df[column].astype(float)

    return X, y, labels, target_names, feature_names
