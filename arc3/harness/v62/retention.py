"""Harness v6.1 item 8: literal UPPER_CASE constants persist across python calls like library functions.

dfranzen's retention (src/inference/utils/retained_functions.py) keeps only top-level functions; a function that reads
a snippet constant is dropped as "unavailable dependencies" (lp85 re-pasted its STAR literal five times). Here a
top-level assignment `NAME = <literal>` with an UPPER_CASE name (at most 4,000 characters of source) is kept while the
name stays bound to that value; it is restored before functions and counts as available for their validation.
Applied to the copy of the module embedded in the sandbox bootstrap (exact anchors, asserted) and, on the host, to
function_signature, which reports a constant as NAME(constant) so the host bookkeeping keeps working.
"""
import ast
import re

from inference.agent import python_tool_sandbox as sb
from inference.agent import tool_agent as ta
from inference.utils import retained_functions as rf

CONST_NAME = re.compile(r'^[A-Z][A-Z0-9_]*$')
MAX_SOURCE = 4000

HELPER = '''

_V61_CONST_NAME = __import__("re").compile(r"^[A-Z][A-Z0-9_]*$")


def _v61_constant(source_or_tree):
    """The name of a retainable constant `NAME = <literal>` (UPPER_CASE, <= 4,000 chars), else None."""
    try:
        tree = ast.parse(source_or_tree) if isinstance(source_or_tree, str) else source_or_tree
    except Exception:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    node = tree.body[0]
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    if not _V61_CONST_NAME.match(node.targets[0].id):
        return None
    try:
        ast.literal_eval(node.value)
    except Exception:
        return None
    return node.targets[0].id

'''

EDITS = [
    # the helper, after the module's imports
    ('import types\n', 'import types\n' + HELPER),
    # prepare_functions: a constant is a record without reads
    ('''            parsed = ast.parse(source)
            if (parsed.body and isinstance(parsed.body[-1], ast.FunctionDef)''',
     '''            parsed = ast.parse(source)
            _const = _v61_constant(parsed)
            if _const is not None and len(source) <= 4000:
                name = _const
                records.pop(name, None)
                records[name] = (source, set())
                continue
            if (parsed.body and isinstance(parsed.body[-1], ast.FunctionDef)'''),
    # prepare_functions: the import-bundle check skips constants
    ('''    for name, (source, _) in list(records.items()):
        _, imports = _source_parts(source)
        if imports.keys() & function_names:''',
     '''    for name, (source, _) in list(records.items()):
        if _v61_constant(source) is not None:
            continue
        _, imports = _source_parts(source)
        if imports.keys() & function_names:'''),
    # collect_functions: constants from previous calls and this snippet, kept while still bound to that literal
    ('''    import_nodes = []
    for source in previous.values():''',
     '''    _v61_consts = {}
    for _v61_name, _v61_src in list(previous.items()):
        if _v61_constant(_v61_src) is not None:
            _v61_consts[_v61_name] = _v61_src
    _v62_prev_consts = set(_v61_consts)
    _v62_changed = []
    for _v61_node in tree.body:
        if isinstance(_v61_node, ast.Assign):
            _v61_seg = ast.get_source_segment(code_text, _v61_node)
            if _v61_seg and len(_v61_seg) <= 4000:
                _v61_name = _v61_constant(_v61_seg)
                if _v61_name is not None:
                    _v61_consts[_v61_name] = _v61_seg
    for _v61_name, _v61_src in list(_v61_consts.items()):
        try:
            _v61_val = ast.literal_eval(ast.parse(_v61_src).body[0].value)
            _v61_keep = (_v61_name not in available and _v61_name not in fresh and _v61_name in namespace
                         and type(namespace[_v61_name]) is type(_v61_val) and namespace[_v61_name] == _v61_val)
        except Exception:
            _v61_keep = False
        if not _v61_keep:
            del _v61_consts[_v61_name]
            if _v61_name in namespace and _v61_name not in available and _v61_name not in fresh:
                _v62_changed.append(_v61_name)
    previous = {k: v for k, v in previous.items() if _v61_constant(v) is None}
    import_nodes = []
    for source in previous.values():'''),
    # v6.2 (b) (run A: tr87, vc33, ft09, dc22, s5i5, sk48): say why an UPPER_CASE name bound to a computed value is not kept
    ('''    return kept, rejected + invalid''',
     '''    _v62_literal = ('only an UPPER_CASE name bound to a literal (a number, string, or tuple/list/dict/set of literals) '
                    'is kept between calls')
    _v62_noted = set()
    for _v62_node in tree.body:
        _v62_names = []
        if isinstance(_v62_node, ast.Assign) and len(_v62_node.targets) == 1:
            _v62_t = _v62_node.targets[0]
            if isinstance(_v62_t, ast.Name):
                _v62_names = [_v62_t.id]
            elif isinstance(_v62_t, (ast.Tuple, ast.List)):      # v6.2 round 3 (A2, tr87): `X0, Y0 = ...`
                _v62_names = [e.id for e in _v62_t.elts if isinstance(e, ast.Name)]
                try:                                   # round 4 (A3 ar25 `AH, AV = 34, 19`, sk48): literals unpacked
                    ast.literal_eval(_v62_node.value)
                    _v62_unpacked = [n for n in _v62_names if _V61_CONST_NAME.match(n) and n in namespace]
                    for _v62_name in _v62_unpacked:
                        _v62_noted.add(_v62_name)
                        rejected.append(_rejection(
                            _v62_name,
                            'an unpacking assignment is not kept between calls; assign each UPPER_CASE constant to a '
                            'literal on its own line (`NAME = <literal>`)',
                            'Recompute it in each call, or return it from a library function.'))
                    _v62_names = []
                except Exception:
                    pass
                _v62_node = ast.Assign(targets=[_v62_t], value=ast.Name(id='_', ctx=ast.Load()))   # never a literal
        # round 3 (A2 wa30): only names the call actually bound
        for _v62_name in [n for n in _v62_names if _V61_CONST_NAME.match(n) and n in namespace]:
            try:
                ast.literal_eval(_v62_node.value)
            except Exception:
                _v62_noted.add(_v62_name)
                rejected.append(_rejection(
                    _v62_name,
                    # round 3 (A2 sc25: a saved D rebound to a computed value read as "constants do not persist")
                    ('it was reassigned to a computed value, so the constant saved earlier was removed; ' + _v62_literal)
                    if _v62_name in _v62_prev_consts else _v62_literal,
                    'Recompute it in each call, or return it from a library function.'))
    # round 3 (A2 m0r0: `CELLS = {}` filled in place was dropped without a note): a literal whose value changed
    for _v62_name in _v62_changed:
        if _v62_name not in _v62_noted:
            rejected.append(_rejection(
                _v62_name,
                'its value changed after the literal assignment (for example it was modified in place), and only an '
                'UPPER_CASE name that still equals its literal at the end of the call is kept between calls',
                'Recompute it in each call, or return it from a library function.'))
    return kept, rejected + invalid'''),
    ('''    kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
    for item in invalid:''',
     '''    candidates.update(_v61_consts)
    kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
    for item in invalid:'''),
    # round 5h (G2, 386 fy09, sx80, ly52, lx52, wy30): a saved program whose name this call bound to another value (a loop
    # variable, a walrus, an assignment) or deleted, without a new top-level def of that name, is kept; the note says so
    # (the generic "definition did not execute, was removed/replaced, ..." rejection dropped it and its dependents)
    ('''    for name in dict.fromkeys([*previous, *fresh]):
        value = namespace.get(name)
''',
     '''    for name in dict.fromkeys([*previous, *fresh]):
        value = namespace.get(name)
        if (name in previous and name not in fresh and name not in available
                and not (type(value) is types.FunctionType and value.__name__ == name)):
            candidates[name] = previous[name]
            rejected.append(_rejection(name, '__v62_kept_rebound__'))
            continue
'''),
    # round 5l (item 110, A6; 392b g508 7 -> 6, ub01 13 -> 10 with the dependants of crop): a new definition of a saved
    # program that is not saved (rebound after the def, an unsupported def, or failed validation) never replaces it; the
    # earlier version is kept and the note says so
    ('''        if not repair_hints and (name in available or type(value) is not types.FunctionType
                                 or value.__name__ != name):
            rejected.append(_rejection(name, 'definition did not execute, was removed/replaced, or conflicts with a provided name'))
            continue''',
     '''        if not repair_hints and (name in available or type(value) is not types.FunctionType
                                 or value.__name__ != name):
            if name in previous and name not in available:
                candidates[name] = previous[name]
                rejected.append(_rejection(name, '__v62_kept_previous__: the new definition did not execute, or its name '
                                                 'was bound to another value'))
                continue
            rejected.append(_rejection(name, 'definition did not execute, was removed/replaced, or conflicts with a provided name'))
            continue'''),
    ('''        if node is None:
            rejected.append(_rejection(name, 'definition was decorated, replaced outside a top-level def, or is unsupported',
                                       'Use a plain top-level def without decorators if you want to reuse it.'))
            continue''',
     '''        if node is None:
            if name in previous:
                candidates[name] = previous[name]
                rejected.append(_rejection(name, '__v62_kept_previous__: the new definition was decorated, made outside a '
                                                 'top-level def, or is unsupported'))
                continue
            rejected.append(_rejection(name, 'definition was decorated, replaced outside a top-level def, or is unsupported',
                                       'Use a plain top-level def without decorators if you want to reuse it.'))
            continue'''),
    ('''    candidates.update(_v61_consts)
    kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
''',
     '''    candidates.update(_v61_consts)
    kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
    _v62_back = [i.get('name') for i in invalid if i.get('name') in previous and i.get('name') in fresh
                 and candidates.get(i.get('name')) != previous[i.get('name')]]
    if _v62_back:
        _v62_why = {i.get('name'): i.get('reason') for i in invalid if i.get('name') in _v62_back}
        for _v62_n in _v62_back:
            candidates[_v62_n] = previous[_v62_n]
        kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
        for _v62_n in _v62_back:
            if _v62_n in kept:
                rejected.append(_rejection(_v62_n, '__v62_kept_previous__: ' + str(_v62_why.get(_v62_n) or 'it did not pass '
                                                                                                        'validation')))
'''),
]


# round 5m (V62_BUILD item 121, 2026-10-08; user: "the length of the game in progress should not affect python call speed"):
# every call is a fresh sandbox process that re-validates and re-executes the whole program library; restore_functions'
# two prepare_functions passes and collect_functions parsed each kept program ~20 times and compiled / disassembled it ~4
# times (200 programs of ~60 lines: ~1.6 s per trivial call). The analysis is a pure function of the source text, so it is
# memoized for the call (exact: same results, same rejection messages; cached dicts and sets are copied on return).
EDITS += [
    ('            parsed = ast.parse(source)\n            _const = _v61_constant(parsed)\n',
     '            parsed = _v62_parse(source)\n            _const = _v61_constant(parsed)\n'),
    ('        import_nodes.extend(ast.parse(source).body[:-1])\n',
     '        import_nodes.extend(_v62_parse(source).body[:-1])\n'),
]

MEMO = '''

# round 5m (item 121): per-call memoization of the library analysis (pure functions of the source text)
_v62_cache = {}


def _v62_cached(kind, key, fn):
    hit = _v62_cache.get((kind, key))
    if hit is None:
        try:
            hit = (True, fn())
        except Exception as exc:
            hit = (False, exc)
        _v62_cache[(kind, key)] = hit
    if not hit[0]:
        raise hit[1].with_traceback(None)
    return hit[1]


def _v62_tree(source):
    """The parse of a source, shared by text up to trailing whitespace (which never changes the tree)."""
    return _v62_cached('tree', source.rstrip(), lambda: ast.parse(source, '<retained>', 'exec'))


def _v62_parse(source):
    """ast.parse(source) for the inline calls: the shared tree when the source parses (the file name only appears in
    syntax errors), else the exact call, so a rejection keeps its message."""
    try:
        return _v62_tree(source)
    except Exception:
        return _v62_cached('parse', source, lambda: ast.parse(source))


def _v62_mode(source, retain_imports):
    """(text key, mode): trailing whitespace never changes the parse; retain_imports only matters with imports."""
    key = source.rstrip()
    try:
        tree = _v62_tree(source)
    except Exception:
        return key, bool(retain_imports)
    return key, (None if len(tree.body) == 1 else bool(retain_imports))


def _v62_source_parts(source, retain_imports):
    """retained_functions._source_parts on the shared tree (same checks, same order)."""
    tree = _v62_tree(source)
    if (not tree.body or not isinstance(tree.body[-1], ast.FunctionDef)
            or (not retain_imports and len(tree.body) != 1)):
        raise ValueError('only top-level function definitions can be retained')
    imports = _import_statements(tree.body[:-1])
    return tree.body[-1], imports


def _source_parts(source, retain_imports=True):
    node, imports = _v62_cached('parts', _v62_mode(source, retain_imports),
                                lambda: _v62_source_parts(source, retain_imports))
    return node, dict(imports)


_v62_function_record_raw = _function_record


def _function_record(source, retain_imports=False):
    name, _text, reads = _v62_cached('record', _v62_mode(source, retain_imports),
                                     lambda: _v62_function_record_raw(source, retain_imports))
    return name, source, set(reads)


_v62_constant_raw = _v61_constant


def _v61_constant(source_or_tree):
    if isinstance(source_or_tree, str):
        try:
            source_or_tree = _v62_parse(source_or_tree)
        except Exception:
            return None                                 # as the original: an unparsable source is no constant
    return _v62_constant_raw(source_or_tree)


# round 5m (item 122, user 10-08 "Let's try the cache"): the analysis carried across calls. Each call is a fresh process,
# so the per-program analysis (a pure function of the program's text) was recomputed every call: one parse, one compile and
# a bytecode walk per kept program. The sandbox now returns each kept program's _function_record result with its text hash;
# the host keeps them and sends them back, and a seeded record skips the compile and the bytecode walk. Only successful
# records are carried; anything not seeded is analysed as before.
import hashlib as _v62_hashlib
_V62_SEED = {}


def _v62_hash(key):
    return _v62_hashlib.sha1(key.encode('utf-8', 'surrogatepass')).hexdigest()


def _v62_seed(seed):
    if isinstance(seed, dict):
        _V62_SEED.update(seed)


def _v62_export_item(source):
    """{'v62': {hash: {'n': statements, 'r': {mode: [name, reads]}}}} for a kept program analysed in this call, else {}."""
    try:
        key = source.rstrip()
        n = len(_v62_tree(source).body)
    except Exception:
        return {}
    rec = {}
    for mode in (None, True, False):
        hit = _v62_cache.get(('record', (key, mode)))
        if hit is not None and hit[0]:
            rec[str(mode)] = [hit[1][0], sorted(hit[1][2])]
    return {'v62': {_v62_hash(key): {'n': n, 'r': rec}}} if rec else {}


_v62_function_record_memo = _function_record


def _function_record(source, retain_imports=False):
    if _V62_SEED:
        key = source.rstrip()
        entry = _V62_SEED.get(_v62_hash(key))
        if isinstance(entry, dict):
            mode = None if entry.get('n') == 1 else bool(retain_imports)
            hit = (entry.get('r') or {}).get(str(mode))
            if (isinstance(hit, list) and len(hit) == 2 and isinstance(hit[0], str)
                    and isinstance(hit[1], list) and all(isinstance(x, str) for x in hit[1])):
                _v62_cache[('record', (key, mode))] = (True, (hit[0], key, set(hit[1])))
                return hit[0], source, set(hit[1])
    return _v62_function_record_memo(source, retain_imports)
'''

# round 5m (item 122): the host's store of the sandbox's per-program analysis, keyed like the sandbox (sha1 of the text up
# to trailing whitespace); a pure function of the text, so shared by every game of the run, and cleared above the cap.
ANALYSIS = {}
ANALYSIS_CAP = 200000
STATS = dict(calls=0, programs=0, seeded=0, harvested=0)
BOOT_SEED = ('    _persistence_enabled = bool(initial.get("persistence_enabled"))\n',
             '    _persistence_enabled = bool(initial.get("persistence_enabled"))\n'
             '    _v62_seed((initial.get("state") or {}).get("v62_analysis"))          # round 5m (item 122)\n')
BOOT_EXPORT = ('                "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],\n',
               '                "keepable_functions": [dict({"name": n, "source": src}, **_v62_export_item(src))\n'
               '                                       for n, src in kept.items()],              # round 5m (item 122)\n')


def _host_hash(source):
    import hashlib
    return hashlib.sha1(source.rstrip().encode('utf-8', 'surrogatepass')).hexdigest()


def _source_text():
    import inspect
    return inspect.getsource(rf)


def install():
    original = _source_text()
    if sb._SANDBOX_BOOTSTRAP.count(original) != 1:
        raise RuntimeError('retention patch: the embedded retained_functions source was not found exactly once')
    patched = original
    for old, new in EDITS:
        if patched.count(old) != 1:
            raise RuntimeError('retention patch: anchor not found exactly once: %r' % old[:70])
        patched = patched.replace(old, new, 1)
    patched = patched.rstrip('\n') + '\n' + MEMO           # round 5m (item 121)
    compile(patched, '<retained_functions v61>', 'exec')
    sb._SANDBOX_BOOTSTRAP = sb._SANDBOX_BOOTSTRAP.replace(original, patched, 1)

    original_signature = rf.function_signature

    def function_signature(source):
        try:
            tree = ast.parse(source)
            if (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign) and len(tree.body[0].targets) == 1
                    and isinstance(tree.body[0].targets[0], ast.Name)
                    and CONST_NAME.match(tree.body[0].targets[0].id)):
                ast.literal_eval(tree.body[0].value)
                return tree.body[0].targets[0].id + '(constant)'
        except Exception:
            pass
        return original_signature(source)

    rf.function_signature = function_signature
    ta.function_signature = function_signature

    # round 5m (item 122): the analysis across calls (bootstrap hooks, host store, wrapper) and the host's signatures
    for old, new in (BOOT_SEED, BOOT_EXPORT):
        if sb._SANDBOX_BOOTSTRAP.count(old) != 1:
            raise RuntimeError('retention patch: bootstrap anchor not found exactly once: %r' % old[:70])
        sb._SANDBOX_BOOTSTRAP = sb._SANDBOX_BOOTSTRAP.replace(old, new, 1)
    outer = ta.run_sandboxed_python

    def run_sandboxed_python(*args, **kwargs):
        kept = kwargs.get('kept_functions')
        if kept:
            seed = {}
            for source in kept:
                h = _host_hash(source)
                entry = ANALYSIS.get(h)
                if entry is not None:
                    seed[h] = entry
            STATS['calls'] += 1
            STATS['programs'] += len(kept)
            STATS['seeded'] += len(seed)
            if seed:
                kwargs['initial_state'] = dict(kwargs.get('initial_state') or {}, v62_analysis=seed)
        result = outer(*args, **kwargs)
        for item in (result or {}).get('keepable_functions') or []:
            if isinstance(item, dict):
                found = item.pop('v62', None)                   # never reaches the host's library bookkeeping
                if isinstance(found, dict):
                    if len(ANALYSIS) > ANALYSIS_CAP:
                        ANALYSIS.clear()
                    ANALYSIS.update(found)
                    STATS['harvested'] += len(found)
        return result

    ta.run_sandboxed_python = run_sandboxed_python
    signature = ta.function_signature
    signatures = {}

    def function_signature_cached(source):
        """The host's signature of a kept program (parsed for every kept program after every call): cached by text."""
        hit = signatures.get(source)
        if hit is None:
            try:
                hit = (True, signature(source))
            except Exception as exc:
                hit = (False, exc)
            if len(signatures) > ANALYSIS_CAP:
                signatures.clear()
            signatures[source] = hit
        if not hit[0]:
            raise hit[1].with_traceback(None)
        return hit[1]

    rf.function_signature = function_signature_cached
    ta.function_signature = function_signature_cached
    return patched
