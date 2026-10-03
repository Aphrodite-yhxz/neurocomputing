"""
daci.adapters
=============
Real frozen-TSFM adapter for Chronos-Bolt (amazon/chronos-bolt-{small,base}). Fits an 8 GB
GPU easily in bf16. Produces a ForecastTrace identical in shape to the surrogate, so every
calibrator + metric applies unchanged.

Internal signal options:
  signal_mode="output"     (default, robust)  -> features from the model's own predictive
                            distribution shape (IQR, inter-decile range, asymmetry, mu-change).
                            Needs no surgery into the model; works on any TSFM.
  signal_mode="embedding"  (full claim)        -> encoder hidden-state of the
                            context (true internal representation). Best-effort; if your Chronos
                            version exposes encoder differently, adjust `_encode` (one method).

Run this on a machine with HuggingFace access + GPU. See SETUP_WINDOWS.md for commands.
"""
import numpy as np

from .forecasters import BaseForecaster, ForecastTrace

_QLEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]   # subset of Chronos-Bolt's trained levels (no interp warning)


class ChronosForecaster(BaseForecaster):
    def __init__(self, model_name="amazon/chronos-bolt-base", context=512, device="cuda",
                 batch=256, signal_mode="output", pooling="last", dtype="bfloat16",
                 qlevels=None):
        import torch
        from chronos import BaseChronosPipeline
        self.name = model_name.split("/")[-1]
        self.context, self.device, self.batch, self.signal_mode = context, device, batch, signal_mode
        self.pooling = pooling        # "last" (sharp, drift) | "mean" (stable, regime ID) | "both"
        td = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.torch = torch
        self.pipe = BaseChronosPipeline.from_pretrained(
            model_name, device_map=device, dtype=td)
        # qlevels: pass a DENSE set (e.g. 0.05..0.95) for CRPS/reliability; default = trained subset
        self.qlevels = np.array(qlevels if qlevels is not None else _QLEVELS)

    # ---- optional: true encoder embedding of a batch of contexts -------------
    def _encode(self, contexts):
        """Return (B, d_model) encoder representation of the context.

        pooling="last" (DEFAULT): the LAST patch's hidden state. Responds sharply to a regime
            change (only ~16 newest points define it). Use for drift/anticipation signals.
        pooling="mean": masked mean over all patches. A STABLE summary of the whole context
            -- good for regime IDENTIFICATION (clustering), bad for detecting abrupt change
            (a 512-step context = ~32 patches, so a fresh shift is diluted ~32x).
        pooling="both": concat(last, mean).
        """
        torch = self.torch
        model = self.pipe.model                       # ChronosBoltModelForForecasting
        L = max(len(c) for c in contexts)
        arr = np.full((len(contexts), L), np.nan, dtype=np.float32)
        for i, c in enumerate(contexts):
            arr[i, L - len(c):] = c                    # left-pad with NaN (encode masks it)
        x = torch.tensor(arr, device=self.device)
        if not hasattr(model, "encode"):
            raise RuntimeError(
                f"{self.name}: this backbone exposes no `encode()`; run it in BLACK-BOX mode "
                f"(signal_mode='output'). The paper's claim is that black-box is enough.")
        with torch.no_grad():
            enc, _loc_scale, _inemb, attn = model.encode(x)   # enc: (B, S, d_model)
        m = attn.unsqueeze(-1).to(enc.dtype)
        mean_p = (enc * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        last_p = enc[:, -1, :]                         # newest patch -> sharp drift response
        if self.pooling == "mean":
            out = mean_p
        elif self.pooling == "both":
            out = torch.cat([last_p, mean_p], dim=-1)
        else:
            out = last_p
        return out.float().cpu().numpy()

    def run(self, series, warmup):
        torch = self.torch
        series = np.asarray(series, float)
        T = len(series)
        t_idx = np.arange(warmup, T - 1)
        C = self.context
        mu = np.empty(len(t_idx)); y_true = series[t_idx + 1]
        Q = np.empty((len(t_idx), len(self.qlevels)))
        emb = [] if self.signal_mode == "embedding" else None

        for b0 in range(0, len(t_idx), self.batch):
            sl = t_idx[b0:b0 + self.batch]
            ctxs = [series[max(0, t + 1 - C):t + 1] for t in sl]
            tens = [torch.tensor(c, dtype=torch.float32) for c in ctxs]
            with torch.no_grad():
                q, mean = self.pipe.predict_quantiles(
                    tens, prediction_length=1,
                    quantile_levels=list(self.qlevels))
            # Chronos-Bolt returns stacked tensors (B, H, n_q); Chronos-2 returns LISTS of
            # (n_variates, H, n_q). Normalize both to (B, n_q) / (B,).
            if isinstance(q, (list, tuple)):
                q = torch.stack([x.reshape(-1, x.shape[-2], x.shape[-1])[0] for x in q])
                mean = torch.stack([x.reshape(-1, x.shape[-1])[0] for x in mean])
            q = q[:, 0, :].float().cpu().numpy()        # (B, n_q)
            Q[b0:b0 + len(sl)] = q
            mu[b0:b0 + len(sl)] = mean[:, 0].float().cpu().numpy()
            if emb is not None:
                emb.append(self._encode(ctxs))

        from scipy.stats import norm
        ql = self.qlevels
        i25 = int(np.argmin(np.abs(ql - 0.25))); i75 = int(np.argmin(np.abs(ql - 0.75)))
        iqr = Q[:, i75] - Q[:, i25]                      # inner interquantile range (nearest avail)
        nf = norm.ppf(ql[i75]) - norm.ppf(ql[i25])       # matching normal factor (1.0488 for .3/.7)
        sigma = np.maximum(iqr / nf, 1e-3 * np.std(series))      # normal-consistent spread
        # output-derived internal signal (model's own uncertainty structure)
        imed = int(np.argmin(np.abs(ql - 0.5)))
        idr = Q[:, -1] - Q[:, 0]                          # outermost inter-decile range
        asym = Q[:, -1] + Q[:, 0] - 2 * Q[:, imed]        # distributional asymmetry
        dmu = np.diff(mu, prepend=mu[0])
        if self.signal_mode == "embedding":
            internal = np.vstack(emb)
        else:
            internal = np.column_stack([sigma, idr, asym, dmu])

        return ForecastTrace(self.name, t_idx, mu, sigma, y_true,
                             internal=internal, quantiles=Q, q_levels=self.qlevels,
                             change_points=None,
                             meta={"model": self.name, "signal_mode": self.signal_mode})
