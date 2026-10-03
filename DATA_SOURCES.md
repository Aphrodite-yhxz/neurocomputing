# Forecast arrays and data provenance

The repository contains 164 forecast/target archives: 124 original one-step
archives (including 32 photovoltaic pairs), 16 development multi-horizon archives,
16 traffic/exchange-rate archives, and eight new weather/air-quality archives.
These are derived predictions plus evaluation targets, not
an unrestricted replacement for the original datasets.

The historical main PV panel consists of eight series and four archived models.
The development multi-horizon panel includes electricity, solar, ETT and explicitly
constructed/synthetic series. Confirmation uses four traffic and four exchange-rate
columns, each with two models. Models and correlated series are not independent
datasets. Exact indices, prefix lengths, target clocks and constructions are described
in the manuscript and preserved protocols.

Primary data sources:

- LSTNet benchmark matrices, including electricity, solar, traffic and exchange rates:
  https://github.com/laiguokun/multivariate-time-series-data
- ETT dataset: https://github.com/zhouhaoyi/ETDataset
- Earlier ERCOT comparison: https://www.ercot.com/gridinfo/load/load_hist
- Earlier GEFCom2014 comparison: Hong et al., International Journal of Forecasting
  32(3), 896–913 (2016). Its full raw data are not part of this repository.

Forecast archives are identified by model/dataset labels in each NPZ file. The paper
reports incomplete checkpoint metadata for the original main panel and possible
benchmark pretraining exposure. Exact local prediction spot checks do not resolve
those limits. No neural checkpoint weights or complete histories of the earlier
source series are redistributed. Exact new-source downloads and four processed
series are included.

`forecast_manifest.json` links original archive hashes to the minimal repacked
archives. Retained arrays were compared exactly; the different ZIP container hashes
do not indicate altered numeric data. Source-specific data/model terms are not
superseded by any code license in this repository. No blanket unrestricted
redistribution permission for third-party data is asserted.
# Additional sources used in the locked confirmation

The exact downloaded files and processed-series hashes are recorded in
`data/strengthening_v9/manifest.json`. Both sources distribute these data under
CC BY 4.0; retain attribution and identify our hourly aggregation, causal input
filling and selected time ranges when reusing the derived series.

- MPI for Biogeochemistry, Jena Beutenberg roof weather, 2026 temperature and
  pressure: [official data and terms](https://weather.bgc-jena.mpg.de/weather_data.html).
  The bundled `weather.zip` is the exact acquisition snapshot; the live URL changes.
- Chen et al., Beijing Multi-Site Air Quality, UCI repository,
  [DOI 10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G). PM2.5 at Aotizhongxin and Dingling.

Four processed series retain missing targets. Only neural input contexts are
causally forward filled; targets are never imputed for calibration or evaluation.
Model IDs, exact checkpoint hashes and inference environments are saved beside
the eight new forecast archives. New source selection does not establish exclusion
from model pretraining.
