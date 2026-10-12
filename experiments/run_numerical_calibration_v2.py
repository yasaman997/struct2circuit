"""Metadata preview by default; v2 numerical phases require persisted authorization."""
from time import process_time
CPU_ORIGIN=process_time()  # before NumPy/SciPy imports
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('development','validation','diagnostics'))
    p.add_argument('--authorization',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'paper/numerical_calibration_protocol_v2.md')
    p.add_argument('--validation-lock',type=Path);p.add_argument('--write-validation-lock',type=Path)
    p.add_argument('--development-output',type=Path);p.add_argument('--validation-output',type=Path)
    p.add_argument('--resource-record',type=Path)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.phase is None:
        from struct2circuit.calibration_v2.config import manifest,verify_protocol
        verify_protocol(a.protocol);print(json.dumps(manifest(),indent=2));return
    if not a.authorization or not a.output:p.error('Official execution requires authorization and a fresh output')
    if a.phase=='validation' and not a.validation_lock:p.error('Validation requires a v2 lock')
    if a.phase=='diagnostics' and not all((a.development_output,a.validation_output,a.resource_record,a.validation_lock)):
        p.error('Diagnostics require both qualified core outputs, resource receipt and lock')
    if a.worker:
        if a.phase=='diagnostics':
            from struct2circuit.calibration_v2.diagnostics import run_diagnostics
            result=run_diagnostics(a.output,development=a.development_output,validation=a.validation_output,
                resource=a.resource_record,lock=a.validation_lock,authorization=a.authorization,protocol=a.protocol,cpu_origin=CPU_ORIGIN)
        else:
            from struct2circuit.calibration_v2.driver import run_phase
            result=run_phase(a.output,phase=a.phase,authorization=a.authorization,protocol=a.protocol,
                lock=a.validation_lock,lock_output=a.write_validation_lock,cpu_origin=CPU_ORIGIN)
        print(result);return
    # The supervisor imports numerical libraries after waiting: child imports,
    # construction, numerical work, reporting and shutdown are all in wait4 CPU.
    child=subprocess.Popen([sys.executable,__file__,*sys.argv[1:],'--worker'])
    _,status,usage=os.wait4(child.pid,0)
    child.returncode=os.waitstatus_to_exitcode(status)
    if child.returncode:raise SystemExit(child.returncode)
    from struct2circuit.calibration_v2.storage import _record,seal_process,create_lock,authorize
    resource=Path(str(a.output)+'.resources.json')
    if not resource.exists():raise SystemExit('Worker did not finalize a resource cutoff; no further execution permitted')
    record=_record(resource)
    previous_cpu=0.
    if a.phase!='development':
        from struct2circuit.calibration_v2.storage import ValidationGate
        if a.phase=='validation':previous_cpu=ValidationGate(a.validation_lock).current_budget()['cpu_seconds']
        else:previous_cpu=_record(a.resource_record)['cpu_seconds']
    publication_origin=process_time()
    receipt=seal_process(resource,process_cpu_seconds=usage.ru_utime+usage.ru_stime,
        supervisor_cpu_seconds=publication_origin-CPU_ORIGIN,prior_cpu=previous_cpu)
    if a.phase=='development' and a.write_validation_lock:
        m=json.loads((a.output/'calibration_manifest.json').read_text())
        if m.get('status')=='QUALIFIED' and m.get('reporting_complete'):
            auth=authorize(a.authorization,a.phase,a.output)
            create_lock(a.write_validation_lock,a.output,receipt,approval=auth['approval_identity'],publication_origin=publication_origin)


if __name__=='__main__':main()
