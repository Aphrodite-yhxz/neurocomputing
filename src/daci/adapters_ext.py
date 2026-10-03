"""
daci.adapters_ext
=================
Additional backbone FAMILIES for the model-agnostic claim (GPU machine, HF access):

  TimesFMForecaster   TimesFM, decoder-only. Supports BOTH package generations:
                        - pip `timesfm` 2.0.x  -> checkpoint google/timesfm-2.0-500m-pytorch
                        - timesfm >= 2.5 (GitHub) -> google/timesfm-2.5-200m-pytorch
                      The adapter auto-detects which API is installed.
  Moirai2Forecaster   Salesforce/moirai-2.0-R-small via uni2ts. **INCOMPATIBLE with
                      Blackwell GPUs (RTX 50xx)**: uni2ts pins torch<2.5, which has no
                      sm_120 kernels. Keep it OUT of the main env. If you want Moirai
                      anyway, make a SEPARATE conda env (cpu-only) and run a reduced
                      grid there — see docs/RUNBOOK_STAGE3.md, Plan B. Three families
                      (enc-dec, enc-only, dec-only) already carry the claim.

Both emit NATIVE QUANTILES -> the same calibrators/metrics apply unchanged, black-box.
Every third-party call sits behind a private method with an "API DRIFT" comment; if a
package moved, you adjust ONE method. ALWAYS run scripts/smoke_backbones.py first.
"""
from pathlib import Path

import numpy as np

from .forecasters import BaseForecaster, ForecastTrace

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _resolve_local(model_name):
    """If <project_root>/models/<repo-basename>/ exists (manually downloaded weights,
    e.g. from the hf-mirror.com web page when programmatic download fails), use it."""
    local = _MODELS_DIR / model_name.split("/")[-1]
    if local.exists():
        print(f"  [adapters_ext] using LOCAL weights: {local}")
        return str(local), True
    return model_name, False


class TimesFMForecaster(BaseForecaster):
    """TimesFM (torch), auto-detecting the installed package generation.

    New API (timesfm >= 2.5, from GitHub):
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(...); m.compile(ForecastConfig(...))
        point, quant = m.forecast(horizon=1, inputs=[...])   # quant: (B, H, 10) = mean+deciles
    Old API (PyPI timesfm 2.0.x — what `pip install timesfm` gives as of 2026-07):
        m = timesfm.TimesFm(hparams=TimesFmHparams(...), checkpoint=TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
        point, quant = m.forecast(inputs, freq=[0]*B)        # quant: (B, H, 10) = mean+deciles
    Both return quantile column layout [mean, q0.1, ..., q0.9].
    """
    def __init__(self, model_name=None, context=512, batch=64, name=None):
        self.context, self.batch = context, batch
        self._load(model_name)
        # columns 1..9 of the quantile tensor = q0.1..q0.9 (col 0 is the mean)
        self.qlevels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self._qcols = [1, 3, 5, 7, 9]     # -> {0.1, 0.3, 0.5, 0.7, 0.9}
        if name:
            self.name = name

    def _load(self, model_name):                       # --- API DRIFT: adjust here ---
        model_name = model_name or "google/timesfm-2.0-500m-pytorch"
        resolved, is_local = _resolve_local(model_name)
        if is_local:
            # LOCAL-FIRST via transformers (>=4.52 ships TimesFmModelForPrediction; the
            # manually downloaded repo is exactly that format: config.json+model.safetensors).
            # This path needs NO network and NO `timesfm` package.
            import torch
            from transformers import TimesFmModelForPrediction
            self._api = "hf"
            self.torch = torch
            self.tfm = (TimesFmModelForPrediction
                        .from_pretrained(resolved, torch_dtype=torch.float32)
                        .to("cuda").eval())
        else:
            import timesfm
            if hasattr(timesfm, "TimesFM_2p5_200M_torch"):        # new-generation API
                self._api = "2.5"
                model_name = "google/timesfm-2.5-200m-pytorch"
                self.tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
                self.tfm.compile(timesfm.ForecastConfig(
                    max_context=max(self.context, 64), max_horizon=64,
                    normalize_inputs=True, use_continuous_quantile_head=True,
                    force_flip_invariance=True, infer_is_positive=False,
                    fix_quantile_crossing=True))
            else:                                                  # old 2.0.x package API
                self._api = "2.0"
                self.tfm = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend="gpu", per_core_batch_size=self.batch,
                        context_len=max(self.context, 32), horizon_len=32,
                        num_layers=50, use_positional_embedding=False),
                    checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_name))
        self.name = model_name.split("/")[-1]

    def _forecast(self, contexts):                     # --- API DRIFT: adjust here ---
        if self._api == "hf":
            torch = self.torch
            past = [torch.tensor(np.asarray(c, np.float32), device="cuda") for c in contexts]
            freq = [0] * len(contexts)
            with torch.no_grad():
                out = self.tfm(past_values=past, freq=freq, return_dict=True)
            point = out.mean_predictions.float().cpu().numpy()       # (B, H)
            quant = out.full_predictions.float().cpu().numpy()       # (B, H, 10)
        elif self._api == "2.5":
            point, quant = self.tfm.forecast(horizon=1, inputs=contexts)
        else:
            point, quant = self.tfm.forecast(contexts, freq=[0] * len(contexts))
        point = np.asarray(point)[:, 0]                          # (B,)
        quant = np.asarray(quant)[:, 0, :]                       # (B, 10) = mean + deciles
        return point, quant[:, self._qcols]                      # (B,), (B, 5)

    def run(self, series, warmup):
        series = np.asarray(series, float); T = len(series)
        t_idx = np.arange(warmup, T - 1); L = len(t_idx)
        mu = np.empty(L); Q = np.empty((L, len(self.qlevels)))
        for b0 in range(0, L, self.batch):
            sl = t_idx[b0:b0 + self.batch]
            ctxs = [series[max(0, t + 1 - self.context):t + 1] for t in sl]
            p, q = self._forecast(ctxs)
            mu[b0:b0 + len(sl)] = p; Q[b0:b0 + len(sl)] = q
        Q = np.sort(Q, axis=1)                          # enforce monotone quantiles
        iqr = Q[:, 3] - Q[:, 1]
        sigma = np.maximum(iqr / 1.0488, 1e-3 * np.std(series))
        return ForecastTrace(self.name, t_idx, mu, sigma, series[t_idx + 1],
                             internal=None, quantiles=Q, q_levels=self.qlevels,
                             meta={"model": self.name, "family": "TimesFM"})


class Moirai2Forecaster(BaseForecaster):
    """Moirai-2.0-R-small (uni2ts). Decoder-only, direct quantile prediction.

    API (verified against uni2ts examples, 2026-07):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        model = Moirai2Forecast(module=Moirai2Module.from_pretrained(
                    "Salesforce/moirai-2.0-R-small"),
                prediction_length=1, context_length=C, target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(batch_size=B)
    Prediction via the GluonTS predictor interface; forecasts expose .quantile(tau).
    """
    def __init__(self, model_name="Salesforce/moirai-2.0-R-small",
                 context=512, batch=64, name=None):
        self.context, self.batch = context, batch
        self.name = name or model_name.split("/")[-1]
        self.qlevels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self._load(model_name)

    def _load(self, model_name):                       # --- API DRIFT: adjust here ---
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        model_name, _ = _resolve_local(model_name)     # models/<name>/ if manually downloaded
        self._model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(model_name),
            prediction_length=1, context_length=self.context, target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        self._predictor = self._model.create_predictor(batch_size=self.batch)

    def _forecast(self, contexts):                     # --- API DRIFT: adjust here ---
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        start = pd.Period("2020-01-01", freq="h")
        ds = ListDataset([{"target": c, "start": start} for c in contexts], freq="h")
        fcs = list(self._predictor.predict(ds))
        Q = np.stack([[f.quantile(t)[0] for t in self.qlevels] for f in fcs])   # (B, 5)
        mu = np.array([f.quantile(0.5)[0] for f in fcs])
        return mu, Q

    def run(self, series, warmup):
        series = np.asarray(series, float); T = len(series)
        t_idx = np.arange(warmup, T - 1); L = len(t_idx)
        mu = np.empty(L); Q = np.empty((L, len(self.qlevels)))
        for b0 in range(0, L, self.batch):
            sl = t_idx[b0:b0 + self.batch]
            ctxs = [series[max(0, t + 1 - self.context):t + 1] for t in sl]
            p, q = self._forecast(ctxs)
            mu[b0:b0 + len(sl)] = p; Q[b0:b0 + len(sl)] = q
        Q = np.sort(Q, axis=1)
        iqr = Q[:, 3] - Q[:, 1]
        sigma = np.maximum(iqr / 1.0488, 1e-3 * np.std(series))
        return ForecastTrace(self.name, t_idx, mu, sigma, series[t_idx + 1],
                             internal=None, quantiles=Q, q_levels=self.qlevels,
                             meta={"model": self.name, "family": "Moirai"})
