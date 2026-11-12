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
    previous = {k: v for k, v in previous.items() if _v61_constant(v) is None}
    import_nodes = []
    for source in previous.values():'''),
    ('''    kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
    for item in invalid:''',
     '''    candidates.update(_v61_consts)
    kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
    for item in invalid:'''),
]


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
    return patched
