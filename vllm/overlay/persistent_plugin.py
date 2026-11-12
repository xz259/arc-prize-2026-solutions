"""Optional profiling wrapper; the original endpoint factory remains authoritative."""
def create_plugin():
    import os
    if os.environ.get('ARC3_GAMEPLAY_PROFILE','0')!='0':
        import gameplay_server_profile
        gameplay_server_profile.install()
    from persistent_endpoint import create_plugin as original
    return original()
