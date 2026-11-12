"""step_emb.py (2026-10-14, Claude fork 5d20b23b; IE 10-14 post-run round, form A of the step-conditioned head): a learned
per-draft-step vector s_k added to the MTP fuse output for steps k >= 2. The shared head otherwise infers its depth only
implicitly (slot conditionals 2-4 ~0.80). s_k is the same H vector on every hyper-connection stream (as the token
embedding enters the fuse), zero-initialised, so a head that gains it starts at exactly its previous function; step 1
never has one (a step-embedding head's step 1 is the plain head). Stored as the parameter 'step_emb' [rows, H], rows =
trained steps - 1, saved in heads as 'mtp.step_emb'; steps beyond the trained rows run without it (K 5 is scored for
reference only). Shared by train_mtp_sp.py (--step-emb) and mtp_eval_val50.py (heads that carry it)."""
import torch
import torch.nn as nn

NAME = 'step_emb'


def install(model, rows, dtype=torch.float32, device=None):
    """Register a zero step_emb [rows, H] on model (idempotent; returns the parameter)."""
    p = getattr(model, NAME, None)
    if p is None:
        p = nn.Parameter(torch.zeros(rows, model.cfg.hidden_size, dtype=dtype, device=device))
        model.register_parameter(NAME, p)
    elif p.shape[0] != rows:
        raise SystemExit('step_emb has %d rows, wanted %d' % (p.shape[0], rows))
    return p


def fuse_k(model, h, tok, k):
    """model.fuse for draft step k (1-based), plus s_k for k >= 2 when the model carries step embeddings."""
    x0 = model.fuse(h, tok)
    se = getattr(model, NAME, None)
    if se is None or k < 2 or k - 2 >= se.shape[0]:
        return x0
    return x0 + se[k - 2].to(x0.dtype).repeat(model.cfg.hc_count)


def head_rows(path):
    """Rows of 'mtp.step_emb' in a head file (safetensors), 0 if it has none."""
    from safetensors import safe_open
    with safe_open(path, 'pt') as f:
        return f.get_slice('mtp.' + NAME).get_shape()[0] if 'mtp.' + NAME in f.keys() else 0
