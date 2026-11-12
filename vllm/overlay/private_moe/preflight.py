"""Cheap source/build/producer admission before any model or native module load."""
import argparse
import json
from pathlib import Path
import socket
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'bootstrap'))
from arc3_private_moe_bootstrap import settings as read_settings
from arc3_private_moe_adapter import sha,require,verify_build,verify_route_eligibility


def main():
    p=argparse.ArgumentParser();p.add_argument('--settings',type=Path,default=HERE/'settings.json');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();c,path=read_settings(args.settings)
    require(socket.gethostname()==c['expected_hostname'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip()==c['expected_boot_id'],'Wrong host/boot')
    require(sha(Path(c['private']['prepared_root'])/'PREPARE_RESULT.json')==c['prepared_sha256'],'Preparation changed')
    for relative,expected in c['router_sources'].items():
        require(sha(Path(c['runtime_root'])/relative)==expected,'Router source changed')
    built,plan,include=verify_build(c['private']);route=verify_route_eligibility(c)
    require(not args.output.exists(),'Fresh preflight output required')
    args.output.write_text(json.dumps(dict(schema='arc3.private-moe-serving.preflight.v1',passed=True,
        settings_sha256=sha(path),prepared_sha256=c['prepared_sha256'],
        build_result_sha256=c['private']['build_result_sha256'],private_library_sha256=c['private']['private_library_sha256'],
        route_eligibility=route,module_name=plan['module_name'],native_loaded=False,model_loaded=False),indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
