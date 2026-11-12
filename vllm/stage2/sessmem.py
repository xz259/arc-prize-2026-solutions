"""stage2_v1 'sessmem' part (2026-10-04 03:05; user ~01:00 "Sounds good" to items 1-2 of the API-server memory fixes,
~02:45 item 3 dropped: the rolling image window for the fallback rebase stays). Host memory of the session service:
  1. A closed session keeps only its bookkeeping. After production's close (pending operations drained, engine
     resources released, state 'closed', audit written), its processed images (parts), stored turn responses
     (operations) and token list are dropped. The session object stays in SessionManager.sessions, so close / status
     calls keep answering 'closed' (status then reports 0 logical tokens; closed_logical_tokens keeps the count).
  2. Stored turn responses: when a new turn starts, completed operations other than the most recent one are dropped, so
     a session holds the in-flight operation and the latest completed one (a retry of the latest request still returns
     its stored response). Older request IDs are forgotten: a stale replay is still refused by the revision check, but
     reusing an old ID with the current revision now starts a new turn (production refused it as a reused ID).
  3. (2026-10-08, the eviction slowdown: profiling run v62a167p's owners trace) A closed session also drops its
     incremental prefix digest (session_service._PrefixDigest keeps a private copy of the whole token list) and its
     finished collector task and output future. backend.close cancels the collector and gathers it with
     return_exceptions=True, which leaves the task's stored CancelledError in place; its traceback keeps the collector's
     frames alive, and with them the last RequestOutput (the request's whole prompt token list). Per closed session
     ~250K + ~150K list entries at 45 min of play; 392b's 300 closed sessions held ~145M entries (1.16 GB of lists).
     A cancelled task's stored error is consumed before the reference is dropped, so no other holder of the task keeps
     it either. status() on a closed session rebuilds an empty digest.
lc67h (25 games, one hour) held 1.82 GB in closed sessions: parts 1.04, 3,158 responses 0.68, token ids 0.09. The
raw-harness logs keep every request and response (DFlash training data comes from them). An error leaves the session as
production left it (counted)."""
import asyncio
import os

PINS = {'session_service': '7ce0eaef072e45354cfae8476719313a401387ea379b3fe9c3ca9c1ab40193e8'}
KEEP_DONE = 1
STATS = dict(closed_freed=0, freed_parts=0, freed_operations=0, freed_token_ids=0, operations_pruned=0, errors=0, last_error=None,
             freed_digest_ids=0, freed_collectors=0, freed_output_futures=0)
# v7.5 B2 (IE; the user 10-23, review Part 2): ARC3_S2_SESSMEM_DROP=1 also drops a closed session's retained structures (the initial
# request, the parse context, the segments, structural positions, image spans and kinds, the prepare profile, the residency ledger and
# the suffix and history registries), each replaced by its empty default; /health and status() of a closed session read only sizes.
DROP = os.environ.get('ARC3_S2_SESSMEM_DROP', '0') == '1'
if DROP:
    STATS.update(drop=True, dropped_state=0)


def _err(exc):
    STATS['errors'] += 1
    STATS['last_error'] = repr(exc)[:300]


def patch_session_service(mod):
    SM = mod.SessionManager
    close_session = SM._close_session
    operation = SM._operation

    async def _close_session(self, session):
        out = await close_session(self, session)
        try:
            STATS['freed_parts'] += len(session.parts)
            STATS['freed_operations'] += len(session.operations)
            STATS['freed_token_ids'] += len(session.token_ids)
            session.closed_logical_tokens = len(session.token_ids)
            session.parts = []
            session.operations = {}
            session.token_ids = []
            digest = session.__dict__.pop('_arc3_token_digest', None)       # item 3 (10-08)
            STATS['freed_digest_ids'] += len(getattr(digest, 'ids', None) or ())
            for name, key in (('collector', 'freed_collectors'), ('output_future', 'freed_output_futures')):
                fut = getattr(session, name, None)
                if fut is not None and fut.done():
                    if fut.cancelled():
                        try:
                            fut.exception()                 # consumes the stored CancelledError (and its traceback)
                        except asyncio.CancelledError:
                            pass
                    setattr(session, name, None)
                    STATS[key] += 1
            if DROP:                                    # v7.5 B2
                session.initial = {}
                session.parse_context = None
                session.segments, session.structural, session.images = [], [], []
                session.image_kinds, session.prepare_profile = {}, {}
                session.rebase_info = None              # IE: a rebase's dropped observation ids (nothing reads them after close)
                for name in ('residency', 'suffix_registry', 'history_registry'):
                    setattr(session, name, type(getattr(session, name))())
                STATS['dropped_state'] += 1
            STATS['closed_freed'] += 1
        except Exception as exc:                        # noqa: BLE001
            _err(exc)
        return out

    async def _operation(self, session, request_id, payload, *, initial):
        try:
            ops = session.operations
            if request_id not in ops:                   # a new turn (a retry keeps everything)
                done = [k for k, op in ops.items() if op['task'].done()]
                for k in done[:len(done) - KEEP_DONE] if KEEP_DONE else done:
                    del ops[k]
                    STATS['operations_pruned'] += 1
        except Exception as exc:                        # noqa: BLE001
            _err(exc)
        return await operation(self, session, request_id, payload, initial=initial)

    SM._close_session = _close_session
    SM._operation = _operation


PATCHERS = {'session_service': patch_session_service}
