"""Exact per-token form of Qwen3_8FlashNextNGramEmbedding.forward_impl for the CPU offload process (ngram_size 3).

2026-10-26 (Claude fork bdfc5402, the engine plan). The model's forward_impl packs the batch into a [num_reqs, max_seq_len]
rectangle (EOS padding), runs the EOS-segment shifts and the n-gram hash on the whole rectangle and then gathers the token
positions: on a mixed step (decode lanes + one long prefill chunk) that is ~num_reqs x the work (19 ms vs ~0.8 ms of CPU for
28 x 5 + 3,956 tokens on Zen 4). Here the same integer arithmetic runs on the gathered positions only. For a token at column c
of its row R = [ngram_context, packed tokens, EOS padding], the model's rule (shifted token valid iff no EOS in R[c - s .. c - 1])
reduces to: shift 1 = R[c - 1]; shift 2 = EOS if R[c - 1] == EOS else R[c - 2]. Graph-padding tokens (beyond
query_start_loc[-1]) keep the model's clamped (last request, column) mapping, so every output row - padding included - equals
the original's bit for bit (test_fasthash.py compares the gathered row ids)."""
import torch


def install(layer):
    """Bind the per-token lookup to one CPU-offload PLE layer instance; returns False (unchanged) if it does not apply."""
    if int(getattr(layer, 'ngram_size', 0)) != 3 or not hasattr(layer, 'ngram_embedding'):
        return False
    original = layer.forward_impl

    def forward_impl(hidden_states, input_ids, query_start_loc, ngram_context, output_buffer=None):
        if output_buffer is None:
            return original(hidden_states, input_ids, query_start_loc, ngram_context)
        return lookup(layer, input_ids, query_start_loc, ngram_context, output_buffer)

    layer.forward_impl = forward_impl
    layer._arc3_fasthash = True
    return True


def lookup(self, input_ids, query_start_loc, ngram_context, output_buffer):
    eos = self.eos_token_id
    input_ids = input_ids.reshape(-1).long()
    qsl = query_start_loc.long()
    num_reqs = qsl.numel() - 1
    num_tokens = input_ids.shape[0]
    if num_tokens > self.positions_buffer.numel():
        raise ValueError(f"PLE received {num_tokens} tokens, but its workspace supports at most {self.positions_buffer.numel()}")
    if num_reqs > self.padded_buffer.shape[0]:
        raise ValueError(f"PLE received {num_reqs} requests, but its workspace supports at most {self.padded_buffer.shape[0]}")
    if num_reqs <= 0:
        raise ValueError("PLE CPU offload requires at least one request")
    lens = qsl[1:] - qsl[:-1]
    max_seq_len = max(1, int(lens.max().item()))
    num_valid = min(int(qsl[-1].item()), num_tokens)   # the original scatters only these into its EOS-filled rows
    positions = self.positions_buffer[:num_tokens]
    req = torch.searchsorted(qsl, positions, right=True) - 1
    req.clamp_(max=num_reqs - 1)
    start = qsl[req]
    col = (positions - start).clamp(0, max_seq_len - 1)
    req_len = lens[req]
    ctx = ngram_context[:num_reqs].to(dtype=torch.long)
    last = max(num_tokens - 1, 0)

    def row(k):
        """R[k] of each token's row (k = column - shift + 2 >= 0): context for k < 2, the request's token, or EOS padding."""
        j = k - 2
        tok = torch.where((j < req_len) & (start + j < num_valid), input_ids[(start + j).clamp(0, last)], eos)
        return torch.where(k < 2, ctx[req, k.clamp(0, 1)], tok)

    c = col + 2
    s0 = row(c)
    s1 = row(c - 1)
    s2 = torch.where(s1 == eos, eos, row(c - 2))
    m = self.layer_multipliers
    mixed2 = torch.bitwise_xor(s0 * m[0], s1 * m[1])
    mixed3 = torch.bitwise_xor(mixed2, s2 * m[2])
    hpn = self.heads_per_ngram
    sizes, offsets = self.ngram_heads_vocab_sizes, self.ngram_heads_offsets
    ids = torch.cat([torch.remainder(mixed2.unsqueeze(-1), sizes[:hpn]) + offsets[:hpn],
                     torch.remainder(mixed3.unsqueeze(-1), sizes[hpn:2 * hpn]) + offsets[hpn:2 * hpn]], dim=-1)
    output = output_buffer[:num_tokens, : self.embedding_dim]
    torch.index_select(self.ngram_embedding.weight, 0, ids.reshape(-1), out=output.reshape(-1, self.head_dim))
    return output


def selfcheck(layer, original, cases=300, seed=0):
    """Load-time equivalence check in the offload process: original vs per-token lookup on random batches (decode, mixed, empty
    requests, graph padding, EOS-heavy), comparing the gathered output rows from the real table bit for bit. Returns
    (cases, mismatches)."""
    g = torch.Generator().manual_seed(seed)
    eos, vocab = layer.eos_token_id, int(getattr(layer, 'unigram_vocab_size', 248320))
    max_t = min(int(layer.positions_buffer.numel()), 4096 + 64)
    max_r = min(int(layer.padded_buffer.shape[0]), 32)
    out_a = torch.zeros(max_t, layer.embedding_dim, dtype=layer.ngram_embedding.weight.dtype)
    out_b = torch.zeros_like(out_a)

    def rint(lo, hi):
        return int(torch.randint(lo, hi + 1, (1,), generator=g))

    bad = done = 0
    for k in range(cases):
        kind = k % 4
        if kind == 0:
            R = rint(1, max_r); lens = [rint(1, 6) for _ in range(R)]
        elif kind == 1:
            R = rint(2, max_r); lens = [5] * R; lens[rint(0, R - 1)] = rint(6, max(6, max_t - 64 - 5 * (R - 1)))
        elif kind == 2:
            R = rint(2, max_r); lens = [rint(0, 6) for _ in range(R)]; lens[0] = max(1, lens[0])
        else:
            R = rint(2, min(20, max_r)); lens = [rint(3, 200) for _ in range(R)]; lens[-1] = rint(0, 2)
        n = min(sum(lens) + (rint(1, 40) if k % 3 else 0), max_t)
        if sum(lens) > n:
            continue
        qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), dtype=torch.int32)
        p_eos = (0.0, 0.05, 0.5)[k % 3]
        ids = torch.randint(0, vocab, (n,), generator=g, dtype=torch.int32)
        ctx = torch.randint(0, vocab, (R, 2), generator=g, dtype=torch.int32)
        if p_eos:
            ids[torch.rand(n, generator=g) < p_eos] = eos
            ctx[torch.rand(R, 2, generator=g) < p_eos] = eos
        original(ids, ids, qsl, ctx, output_buffer=out_a)
        lookup(layer, ids, qsl, ctx, out_b)
        done += 1
        if not torch.equal(out_a[:n].view(torch.int16), out_b[:n].view(torch.int16)):
            bad += 1
    return done, bad
