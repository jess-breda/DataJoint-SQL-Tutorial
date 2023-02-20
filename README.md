# DataJoint-SQL-Tutorial

Tutorial notebooks on how to setup, access and utilize SQL tables via DataJoint in the Brody Lab.


## Set up

1. Follow setup information fom the Brody Lab DataJoint Pipeline [repository](https://github.com/Brody-Lab/bl_pipeline_python)

2. Clone this repository 

```
git clone https://github.com/jess-breda/DataJoint-SQL-Tutorial.git
```

3. Activate environment created in step 1 & install additional library
```
conda activate <env name>
pip install seaborn
```

4. Use your preferred IDE to work through the notebooks.

## Overview

**`1-dj-sql-overview`** : live coding of commands to access DataJoint information for the Brody lab.
* supporting files: `fetch_water.py`

**`2-trial-information`**: where and how to access trial-by-trial behavior data
* supporting files: `dj_utils.py`

**`3-training-progress`**: example implementation of dj tables to track animal performance
* supporting files: `pd_to_df.py`, `performance_plots.py`







