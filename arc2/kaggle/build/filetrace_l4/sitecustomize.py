"""filetrace_l4/sitecustomize.py (2026-10-31, ARC2 - main): the ARC-3 Kaggle port's file tracer (inference_stack/kaggle_upload/
filetrace/sitecustomize.py 7da53f0d, unchanged below this docstring except two lines) for the L4 production path, where the tracer
sits before bootstrap_l4prod on the path. bootstrap_l4prod's pass-through (no L4_NEXT_SITECUSTOMIZE: the launcher and other
non-server processes) runs the first other sitecustomize on sys.path, which is this one, as run_name 'l4prod_passthrough': on that
re-entry the hook is already installed (this file ran first), so it is not installed again, and the chain skips this directory and
every bootstrap_l4prod* directory, which continues the chain exactly where bootstrap_l4prod's pass-through would have without the
tracer (without the skip, the two pass-throughs would call each other forever). The original docstring follows.
sitecustomize.py (2026-10-24, Claude main session 099768dc; the Kaggle port's file tracer, KAGGLE_PORT_PLAN_2026-10-24.md): when
ARC3_FILETRACE_DIR is set, every Python process that starts with this directory on sys.path (PYTHONPATH, inherited by the processes
it spawns) records, once per path, the files it opens, the modules it imports (with their files), the libraries it loads through
ctypes and the programs it starts, in $ARC3_FILETRACE_DIR/trace-<pid>.txt (the first line is the process's argv). One traced launch
of the server and the harness yields the exact file set the Kaggle code bundle must carry. Inert when the variable is unset. Any later
sitecustomize on sys.path still runs (this module executes the next one it finds)."""
import os
import sys

_REENTRY = __name__ == 'l4prod_passthrough'
_D = os.environ.get('ARC3_FILETRACE_DIR')
if _D and not _REENTRY:
    try:
        os.makedirs(_D, exist_ok=True)
        _seen = set()
        _fh = open(os.path.join(_D, 'trace-%d.txt' % os.getpid()), 'a', buffering=1)
        _fh.write('#argv %r ppid %d\n' % (sys.argv, os.getppid()))

        def _rec(kind, path):
            try:
                if isinstance(path, int) or path is None:
                    return
                if isinstance(path, bytes):
                    path = path.decode('utf-8', 'replace')
                path = os.fspath(path)
                if not isinstance(path, str):
                    return
                if not os.path.isabs(path) and not (kind == 'exec' and os.sep not in path):   # a bare program name stays as is
                    path = os.path.abspath(path)
                key = (kind, path)
                if key in _seen:
                    return
                _seen.add(key)
                _fh.write('%s %s\n' % (kind, path))
            except Exception:  # noqa: BLE001
                pass

        def _hook(event, args):
            if event == 'open':
                _rec('open', args[0] if args else None)
            elif event == 'import':
                if len(args) > 1 and args[1]:
                    _rec('import', args[1])
            elif event == 'ctypes.dlopen':
                _rec('dlopen', args[0] if args else None)
            elif event in ('subprocess.Popen', 'os.exec', 'os.posix_spawn'):
                exe = args[0] if args else None
                _rec('exec', exe)

        sys.addaudithook(_hook)
    except Exception:  # noqa: BLE001
        pass

# chain to the next sitecustomize on sys.path, if any (this file shadows it)
try:
    _here = os.path.dirname(os.path.abspath(__file__))
    for _p in sys.path:
        if not _p or os.path.abspath(_p) == _here or (_REENTRY and os.path.basename(os.path.abspath(_p)).startswith('bootstrap_l4prod')):
            continue
        _f = os.path.join(_p, 'sitecustomize.py')
        if os.path.isfile(_f):
            exec(compile(open(_f).read(), _f, 'exec'), {'__name__': 'sitecustomize_chained', '__file__': _f})
            break
except Exception:  # noqa: BLE001
    pass
