tstoy04 separate series csv files
---------------------------------

Each separate csv file in `tstoy04/SeparateSeries/` corresponds to one of 42 series in the tstoy04 data subset (spaning 2020--2022). The series are numbered 01--42, the ordering matches the rows in tstoy04_series.csv. In tstoy04_series.csv the TreeNet series names are matched to the TreeNet site names. In tstoy04_sites.csv each row is a site, with columns being lat/lon and projected coordinates (following CH1903+).

For each separate series csv file cols are (ts, twd, pr, at, ws, dp, sr, lr), where:
* ts = daily time stamp, from 2020-01-01 till 2022-12-31
* twd = target TWD_norm variable, unprocessed
* pr = precipitation in kg/m^2
* at = air temperature in °C
* ws = wind speed in m/S
* dp = dew point in °C
* sr = net short wave radiation flux in W/m^2
* lr = net long wave radiation flux in W/m^2
