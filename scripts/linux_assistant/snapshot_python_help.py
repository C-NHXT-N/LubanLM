"""Pin local official interpreter/pip help; no dependency or environment changes."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime,timezone


def main():
    workspace=Path(__file__).resolve().parents[3]
    output=workspace/'provenance/linux-assistant-upstream/python-installed-help'
    files={
        'python-help.txt':subprocess.check_output([sys.executable,'-X','utf8','--help']),
        'pip-help.txt':subprocess.check_output([sys.executable,'-X','utf8','-m','pip','--help']),
        'pip-show-help.txt':subprocess.check_output([sys.executable,'-X','utf8','-m','pip','show','--help']),
        'sys-module-doc.txt':sys.__doc__.encode('utf8'),
        'PYTHON-LICENSE.txt':(Path(sys.base_prefix)/'LICENSE_PYTHON.txt').read_bytes(),
    }
    dist=importlib.metadata.distribution('pip')
    files['pip-LICENSE.txt']=Path(dist.locate_file('pip-'+dist.version+'.dist-info/licenses/LICENSE.txt')).read_bytes()
    files['pip-METADATA.txt']=dist.read_text('METADATA').encode('utf8')
    output.mkdir(parents=True,exist_ok=True)
    for name,content in files.items():
        path=output/name
        if path.exists() and path.read_bytes()!=content:
            raise ValueError('Pinned help changed: '+name)
        path.write_bytes(content)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
              'python_version':sys.version,'pip_version':dist.version,'executable':sys.executable,
              'platform':sys.platform,'scope':'Installed official CLI/module documentation; platform-specific notes remain Windows-specific.',
              'files':{name:hashlib.sha256(content).hexdigest() for name,content in files.items()},
              'content_review':'pending_independent_review','package_installation_performed':False}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
