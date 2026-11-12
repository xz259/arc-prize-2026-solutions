"""Opt-in process-start import interception; no inference package imported here."""
import os
if os.environ.get('ARC3_PRIVATE_MOE_SERVING') == '1':
    try:
        from arc3_private_moe_bootstrap import install
        install(os.environ['ARC3_PRIVATE_MOE_SETTINGS'])
    except BaseException as error:
        # site.py otherwise logs and ignores ordinary startup exceptions.
        raise SystemExit('Private MoE startup failed: ' + repr(error)) from error
