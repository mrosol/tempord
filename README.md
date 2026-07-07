# tempord

`tempord` is a  Python package for computing **temporal order** between pairs of time series signals based on paper:
```
M. Młyńczak, "Temporal orders and causal vector for physiological data analysis," 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada, 2020, pp. 750-753, doi: 10.1109/EMBC44109.2020.9176842.
```

---

## 📦 Installation

You can install `tempord` from PyPI or clone the repository for development.

### Option 1: Install from PyPI

```bash
pip install tempord
```

### Option 2: Clone and install from source

1. **Clone the repository**
   ```bash
   git clone git@github.com:mrosol/tempord.git
   cd tempord
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   # .\.venv\Scripts\activate  # Windows
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage

Import the main function from the `tempord` module and call it with a `pandas.DataFrame` containing your signals as columns.

### Method: LM (linear regression, adjusted R²)

Measures how well one shifted signal predicts the other using adjusted R². Higher values indicate stronger linear dependence at a given shift.

```python
from tempord import tempord
import pandas as pd

df = pd.DataFrame({'s1': signal1, 's2': signal2})

results = tempord(
    df,
    method="LM",
    modality="raw",              # use raw signal values
    thr=0.7,                     # skip time points where max R² < 0.7
    scaling=1,                   # 0=no scaling, 1=min-max, 2=z-score
    sig_length=10,               # window length in seconds
    max_shift_seconds=(-1, 1),   # backward/forward range in seconds
    fs=25,                       # sampling frequency (Hz)
    make_figure=True
)
```

### Method: TD (time-series distance)

Measures dissimilarity between shifted windows using a distance metric. Lower values indicate greater similarity at a given shift. The `td_type` argument selects the metric.

Supported metrics: `"euclidean"`, `"manhattan"`, `"chebyshev"`, `"correlation"`, `"cosine"`. A custom callable with signature `f(a, b) -> float` is also accepted.

```python
results = tempord(
    df,
    method="TD",
    modality="raw",
    td_type="euclidean",         # distance metric
    thr=5.0,                     # skip time points where min distance > 5.0
    scaling=1,
    sig_length=10,
    max_shift_seconds=(-1, 1),
    fs=25,
    make_figure=True
)
```

### Modality: phase

Setting `modality="phase"` converts each signal to its instantaneous phase before computing temporal order. Continuous signals use the Hilbert transform (zero-mean centred internally); binary signals use a cycle-linear phase. This is useful for oscillatory or binary rhythmic signals where amplitude is not informative.

```python
results = tempord(
    df,
    method="LM",
    modality="phase",            # convert to instantaneous phase first
    thr=0.5,
    scaling=0,                   # phase signals typically need no additional scaling
    sig_length=10,
    max_shift_seconds=(-1, 1),
    fs=25,
    make_figure=True
)
```

### Reading the results

`results` is a dict keyed by signal-pair tuples (e.g. `('s1', 's2')`). Each value contains:

| Key | Content |
|-----|---------|
| `"Tempord"` | `DataFrame` — parameter matrix (rows: shifts, columns: time points) |
| `"Best"` | `DataFrame` — shift at the extremum per time point (after threshold filtering) |
| `"Fig"` | `matplotlib.Figure` — heatmap with the extremum curve overlaid, or `None` |

```python
pair_result = results[('s1', 's2')]
fig = pair_result["Fig"]           # matplotlib figure
best_shifts = pair_result["Best"]  # DataFrame with columns "Points" and "Best"
param_matrix = pair_result["Tempord"]
```

### Plotting and post-processing

The module provides helper functions used internally:
* `get_causal_vector` – extracts the shift at the extremum (max for LM, min for TD) for each time point, subject to the threshold.
* `make_plot` – draws a heatmap of the parameter matrix and overlays the extremum curve.

---

## 📌 Notes

* Input signals must be numeric and of equal length. Windows or shifts that fall outside signal boundaries are skipped.
* For `method="LM"`, the threshold applies to adjusted R² (higher = stronger correlation). For `method="TD"`, the threshold applies to the minimum distance (lower = more similar). Pass `thr=-1` to disable thresholding entirely.
* `modality="phase"` is most meaningful for oscillatory or binary rhythmic signals. For aperiodic signals the Hilbert-based phase may be unreliable.

---

## Contact

mail: maciej.rosol@pw.edu.pl