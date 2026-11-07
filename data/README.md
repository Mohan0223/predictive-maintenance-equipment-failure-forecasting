# Dataset provenance

## AI4I 2020 Predictive Maintenance Dataset

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i)
- Dataset citation: Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Project file: `data/ai4i2020.csv`
- Records: 10,000
- Nature of data: synthetic data designed to reflect industrial predictive-maintenance conditions; it is not telemetry collected from a deployed factory machine.

The project stages the source CSV at `data/ai4i2020.csv`. If the file is missing, run `python src/download_data.py` to fetch and validate it from UCI. Use `python src/download_data.py --force` to download a fresh copy.

## Scope and limitations

The file contains air and process temperature, rotational speed, torque, and tool wear. It does not contain vibration or pressure readings. We will not invent those signals or present them as observed data. The model and dashboard will use the fields supported by this dataset unless a separately sourced dataset is added later.

`Machine failure` is the binary outcome for a record. This is suitable for classifying the recorded operating condition, but the data does not define a prediction horizon such as “failure within the next 24 hours” or a remaining-useful-life target.
