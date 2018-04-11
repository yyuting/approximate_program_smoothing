import os
import hashlib
import multiprocessing
import subprocess

def system_parallel(cmdL, nproc=None, verbose=True):
    """
    Run a list of commands (each a string) via the shell using GNU parallel with nproc processes, return all outputs in a single str instance.
    """
    if nproc is None:
        nproc = multiprocessing.cpu_count()
    sh_filename = '_run_parallel_' + hashlib.md5('\n'.join(cmdL).encode('utf-8')).hexdigest()
    with open(sh_filename, 'wt') as f:
        f.write('\n'.join(cmdL))
    out = subprocess.check_output('parallel -j%d %s--keep-order < %s' % (nproc, '--verbose ' if verbose else '', sh_filename), shell=True)
    out = out.decode('utf-8')
    if verbose:
        print('-'*80)
        print('system_parallel output:')
        print('-'*80)
        print(out)
    os.remove(sh_filename)
    return out


