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

```python
from tempord import tempord
import pandas as pd

# example DataFrame with two signals
df = pd.DataFrame({'s1': signal1, 's2': signal2})

results = tempord(
    df,
    method="LM",                # or "TD"
    thr=0.7,                      # threshold (>=0 to apply)
    scaling=1,                    # 0=no scaling, 1=min-max, 2=z-score
    sig_length=10,                # window length in seconds
    max_shift_seconds=(-1, 1),    # backward/forward range in seconds
    fs=25,                        # sampling frequency (Hz)
    make_figure=True              # generate a matplotlib Figure
)

# `results` is a dict keyed by signal‑pair tuples (e.g. ('s1','s2')).
# Each value contains:
# - "Tempord": DataFrame of parameter values for each point/shift
# - "Max": DataFrame with the shift giving the maximum parameter per point
# - "Fig": matplotlib figure object (or None)
```

### Plotting and post‑processing

The module provides helper functions used internally:
* `get_causal_vector` – extracts the shift of maximum parameter values.
* `make_plot` – draws a heatmap of the parameter matrix and overlays the maxima curve.

---

## 📌 Notes

* Input signals should be numeric and of equal length.  Windowing and shifting that fall outside the signal boundaries are skipped quietly.

---

## Contact

mail: maciej.rosol@pw.edu.pl