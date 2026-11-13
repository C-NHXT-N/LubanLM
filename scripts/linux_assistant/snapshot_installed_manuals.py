"""Copy selected installed Linux documentation, never execute its commands.

Run inside the existing Ubuntu WSL distribution. This supplements provenance;
it does not create question-answer data or approve source content.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone

PAGES={
    '1':('arch','uname','nproc','pwd','readlink','realpath','env','printenv','id',
         'groups','whoami','who','date','stat','df','du','ls','pathchk','head',
         'tail','sort','uniq','cut','paste','join','comm','tr','wc','od','cksum',
         'sha256sum','sha512sum','timeout','test','mktemp','tee','printf','cat',
         'cp','install','ln','mkdir','chmod','chown','find','xargs','grep','sed',
         'tar','gzip','bash','getconf','free','ps','pgrep','vmstat','uptime',
         'top','flock','lscpu','lsblk','findmnt','lslocks','systemctl','journalctl'),
    '5':('proc','sysfs','os-release','fstab','systemd.service','systemd.exec',
         'systemd.resource-control'),
    '8':('ip','ss','mount','swapon','sysctl','lsblk','findmnt'),
}


def digest(data):return hashlib.sha256(data).hexdigest()


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Pinned documentation changed: '+str(path))
    else:path.write_bytes(data)


def main():
    workspace=Path(__file__).resolve().parents[3]
    output=workspace/'provenance/linux-assistant-upstream/installed-linux-manuals'
    if (output/'manifest.json').exists():
        manifest=json.loads((output/'manifest.json').read_text('utf8'))
        for row in manifest['files']:
            if digest((output/row['file']).read_bytes())!=row['sha256']:
                raise ValueError('Pinned manual changed: '+row['file'])
        print(json.dumps({'already_pinned':True,'pages':len(manifest['files'])}));return
    files=[];missing=[];packages={}
    for section,names in PAGES.items():
        for name in names:
            source=Path('/usr/share/man')/('man'+section)/(name+'.'+section+'.gz')
            if not source.is_file():missing.append(str(source));continue
            resolved=source.resolve()
            content=gzip.decompress(resolved.read_bytes())
            if len(content)<80 or content.lstrip().startswith(b'.so '):
                missing.append(str(source)+' (short/roff include; not standalone)');continue
            owner=subprocess.check_output(['dpkg-query','-S',str(source)],text=True).split(': ',1)[0]
            version=subprocess.check_output(['dpkg-query','-W','-f=${Version}',owner],text=True)
            relative=f'{name}.{section}'
            write_once(output/relative,content)
            files.append({'file':relative,'sha256':digest(content),'source_path':str(source),
                          'resolved_source_path':str(resolved),'package':owner,'package_version':version,
                          'source_atom_id':f'installed-manual:{owner}:{version}:{relative}',
                          'content_review':'not_performed','question_answer_records':0})
            if owner not in packages:
                copyright_file=Path('/usr/share/doc')/owner.split(':',1)[0]/'copyright'
                record={'version':version,'copyright_file_available':copyright_file.is_file(),
                        'license_review':'pending_independent_per-page_review'}
                if copyright_file.is_file():
                    data=copyright_file.read_bytes();relative_license='licenses/'+owner.replace(':','_')+'-copyright'
                    write_once(output/relative_license,data)
                    record.update(file=relative_license,sha256=digest(data))
                packages[owner]=record
    for name in ('GPL-1','GPL-2','GPL-3','LGPL-2.1','GFDL-1.3'):
        path=Path('/usr/share/common-licenses')/name
        if path.is_file():write_once(output/'licenses'/name,path.read_bytes())
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
              'source':'Ubuntu-22.04 WSL installed package documentation',
              'os_release':Path('/etc/os-release').read_text('utf8'),
              'files':files,'missing_or_nonstandalone':missing,'packages':packages,
              'dataset_approval':False,'commands_executed_for_behavior_verification':False}
    write_once(output/'manifest.json',(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps({'pages':len(files),'missing':len(missing),'packages':list(packages)},ensure_ascii=False))


if __name__=='__main__':main()
