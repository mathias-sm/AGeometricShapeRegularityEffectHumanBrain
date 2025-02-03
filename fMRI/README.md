# Behavior

`cd 00_behavior ; make` should generate all the stats and plots provided a
working installation of R is available.

# Analyses

For analyses, the subfolder contain numbered scripts to run in order to
generate the intermediary processed data, and then the figures. Most of the
script take arguments for the subject and possibly the task/model; typically
one would run the following:

```bash
python <scripy>.py sub-204 category
```
