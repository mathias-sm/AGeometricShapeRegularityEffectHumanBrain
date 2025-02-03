# Behavior

`cd 00_behavior ; make` should generate all the stats and plots provided a
working installation of R is available.

# Preprocessing

Follow instructions provided at
[https://mne.tools/mne-bids-pipeline/1.5/getting_started/basic_usage.html#create-a-configuration-file](https://mne.tools/mne-bids-pipeline/1.5/getting_started/basic_usage.html#create-a-configuration-file)
to run the preprocessing. Typically this will involve running the following
command, after setting up `mne-bids-pipeline`

```python
mne_bids_pipeline --config=POGS_MEG_config.py
```

Then you should add metada to the trials with `python add_metadata.py`

# Analyses

For analyses, the subfolder contain numbered scripts to run in order to
generate the intermediary processed data, and then the figures.
