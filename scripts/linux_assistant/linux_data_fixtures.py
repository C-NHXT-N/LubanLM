"""Isolated Linux reference-behavior checks; never execute commands from dataset text.

Only fixed authored commands below run. All writes/child processes belong to a
fresh /tmp/lubanlm-data-* directory. No installs, mounts, proxy or board changes.
"""
from __future__ import annotations
import argparse
import codecs
import errno
import gzip
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile

def main(output):
    results=[]
    def test(name,fn):
        try:detail=fn();results.append({'name':name,'passed':True,'detail':detail})
        except Exception as error:results.append({'name':name,'passed':False,'error':type(error).__name__+': '+str(error)})
    def expect(condition,message):
        if not condition:raise AssertionError(message)
    with tempfile.TemporaryDirectory(prefix='lubanlm-data-') as folder:
        root=Path(folder).resolve()
        if root.parent!=Path('/tmp') or not root.name.startswith('lubanlm-data-'):
            raise ValueError('Unexpected fixture directory; no file operations allowed')
        env={**os.environ,'LC_ALL':'C','TMPDIR':str(root),'MODEL_DIR':''}
        def run(command):return subprocess.run(['bash','--noprofile','--norc','-c',command],cwd=root,env=env,capture_output=True,text=True,timeout=10)
        def py(command):return 'python3 -c '+shlex.quote(command)
        def shell(command,stdout=None,code=0):
            p=run(command);expect(p.returncode==code,f'expected status {code}, got {p.returncode}: {p.stderr[:200]}')
            if stdout is not None:expect(p.stdout==stdout,repr(p.stdout)+' != '+repr(stdout))
            return {'command':command,'status':p.returncode,'stdout':p.stdout[:500]}
        (root/'lines').write_text('a\nb\nc\n');(root/'short').write_text('a\nb')
        (root/'.hidden').write_text('x');(root/'-notes').write_text('note\n')
        (root/'grep.log').write_text('axb\na.b\n')
        (root/'source').mkdir();(root/'source/item').write_text('x')
        (root/'dest').mkdir();(root/'flat').mkdir()
        (root/'names').write_text(' one\\two \nlast')
        (root/'syms').mkdir();(root/'weights').mkdir();(root/'weights/a.bin').write_text('x')
        (root/'syms/link').symlink_to('../weights/a.bin')
        test('kernel-release',lambda:shell('uname -r',os.uname().release+'\n'))
        test('machine-architecture',lambda:shell('uname -m',os.uname().machine+'\n'))
        test('effective-uid',lambda:shell('id -u',str(os.geteuid())+'\n'))
        test('real-uid',lambda:shell('id -ru',str(os.getuid())+'\n'))
        test('all-groups',lambda:expect(set(map(int,run('id -G').stdout.split()))==set([os.getegid(),*os.getgroups()]),'group mismatch'))
        test('cwd-relative-path',lambda:shell('mkdir -p location/models; cd location; pwd; test -d models',str(root/'location')+'\n'))
        test('hidden-file-listing',lambda:expect('.hidden' not in run('ls').stdout.splitlines() and '.hidden' in run('ls -a').stdout.splitlines(),'hidden listing wrong'))
        test('mkdir-parents-existing',lambda:shell('mkdir -p a/b; mkdir -p a/b; test -d a/b'))
        test('mkdir-middle-file-fails',lambda:expect(run('mkdir -p short/child').returncode!=0,'ordinary file treated as directory'))
        test('head-short-no-padding',lambda:shell('head -n 20 short','a\nb'))
        test('head-limited',lambda:shell('head -n 2 lines','a\nb\n'))
        test('cp-archive-existing-destination',lambda:shell('cp -a source dest; test -f dest/source/item'))
        test('cp-directory-contents',lambda:shell('cp -a source/. flat/; test -f flat/item; test ! -d flat/source'))
        test('quoted-find-pattern',lambda:shell('mkdir -p search/sub; touch search/a.rkllm search/sub/b.rkllm; find search -type f -name "*.rkllm" | LC_ALL=C sort','search/a.rkllm\nsearch/sub/b.rkllm\n'))
        test('find-depth-limit',lambda:shell('find search -maxdepth 1 -type f -name "*.rkllm"','search/a.rkllm\n'))
        test('grep-fixed-not-regex',lambda:shell('grep -n -F -- "a.b" grep.log','2:a.b\n'))
        test('grep-no-match-status',lambda:shell('grep -F -- missing grep.log','',1))
        test('grep-error-status',lambda:shell('grep -F -- text absent-file','',2))
        test('read-backslash-spaces-last-line',lambda:shell('while IFS= read -r line || [ -n "$line" ]; do printf "%s\\n" "$line"; done < names',' one\\two \nlast\n'))
        test('pipe-default-status',lambda:shell('false | cat; printf "%s\\n" "$?"','0\n'))
        test('pipefail-status',lambda:shell('set -o pipefail; (exit 2) | cat; printf "%s\\n" "$?"','2\n'))
        test('leading-dash-file',lambda:shell('cat -- -notes','note\n'))
        test('mktemp-creates-directory',lambda:shell('d=$(mktemp -d); test -d "$d"'))
        test('mktemp-u-does-not-create',lambda:shell('d=$(mktemp -u); test ! -e "$d"'))
        test('cut-delimited-field',lambda:shell('printf "%s\\n" "model:ready:4" | cut -d: -f2','ready\n'))
        test('xargs-empty-no-run',lambda:shell('printf "" | xargs -r printf marker',''))
        test('xargs-empty-default-run',lambda:shell('printf "" | xargs printf marker','marker'))
        test('required-empty-rejected',lambda:expect(run(': "${MODEL_DIR:?missing}"').returncode!=0,'empty path accepted'))
        test('required-nonempty-not-directory',lambda:shell('MODEL_DIR=/definitely/absent/lubanlm-fixture; : "${MODEL_DIR:?missing}"; test ! -d "$MODEL_DIR"'))
        test('command-scoped-environment',lambda:shell("MODE=old; MODE=new bash -c 'printf \"%s\\n\" \"$MODE\"'; printf '%s\\n' \"$MODE\"",'new\nold\n'))
        test('hard-link-shared-inode',lambda:shell('printf first > hard-a; ln hard-a hard-b; printf second > hard-b; cmp hard-a hard-b; cat hard-a','second'))
        test('relative-symlink-target',lambda:shell('readlink -f syms/link',str(root/'weights/a.bin')+'\n'))
        test('realpath-equivalent',lambda:shell('realpath syms/link',str(root/'weights/a.bin')+'\n'))
        test('cmp-same',lambda:shell('cmp lines lines',''))
        test('cmp-different',lambda:shell('cmp -s lines short','',1))
        test('cmp-error-not-difference',lambda:expect(run('cmp lines absent-file').returncode==2,'cmp error not status2'))
        test('tee-overwrite-append',lambda:shell('printf old > tee.log; printf new | tee tee.log >/dev/null; printf plus | tee -a tee.log >/dev/null; cat tee.log','newplus'))
        test('safe-printf-data',lambda:shell("value='%s\\n'; printf '%s\\n' \"$value\"",'%s\\n\n'))
        test('command-v-function',lambda:shell('local_demo(){ :; }; command -v local_demo','local_demo\n'))
        test('jobs-current-shell',lambda:expect('sleep' in run('sleep 0.1 & jobs -l; wait').stdout,'missing job'))
        test('umask-clears-not-adds',lambda:shell('umask 022; '+py('import os; fd=os.open("mask-a",os.O_CREAT|os.O_WRONLY,0o666); os.close(fd); fd=os.open("mask-b",os.O_CREAT|os.O_WRONLY,0o600); os.close(fd)')+'; stat -c %a mask-a mask-b','644\n600\n'))
        test('stat-includes-execute-bit',lambda:shell('chmod 641 lines; stat -c %a lines','641\n'))
        test('stat-logical-size',lambda:shell('stat -c %s short','3\n'))
        test('sha256-manifest-positive',lambda:shell('sha256sum lines > SHA256SUMS; sha256sum -c SHA256SUMS','lines: OK\n'))
        test('sha256-manifest-negative',lambda:expect(run('printf change >> lines; sha256sum -c SHA256SUMS').returncode!=0,'changed bytes accepted'))
        test('sha256-invalid-r-rejected',lambda:expect(run('sha256sum -r -c SHA256SUMS').returncode!=0,'unsupported flag accepted'))
        test('wc-counts-newline-not-fragments',lambda:shell('wc -l < short','1\n'))
        test('tr-removes-cr-copy',lambda:shell('printf "a\\r\\nb\\r\\n" > crlf; tr -d "\\r" < crlf > cleaned; cat cleaned','a\nb\n'))
        test('sed-first-only',lambda:shell('printf "a a a\\n" | sed "s/a/b/"','b a a\n'))
        test('basename',lambda:shell('basename /opt/chat/model.bin','model.bin\n'))
        test('uniq-only-adjacent',lambda:shell('printf "a\\nb\\na\\n" | uniq','a\nb\na\n'))
        test('directory-versus-file-test',lambda:shell('test -d source; test ! -f source'))
        test('comm-sorted-common',lambda:shell('printf "a\\nb\\n" > left; printf "b\\nc\\n" > right; comm -12 left right','b\n'))
        test('stdout-pipe-not-tty',lambda:shell('test -t 1','',1))
        test('getconf-pagesize',lambda:shell('getconf PAGESIZE',str(os.sysconf('SC_PAGESIZE'))+'\n'))
        test('getconf-argmax',lambda:expect(int(run('getconf ARG_MAX').stdout)>0,'ARG_MAX invalid'))
        test('proc-commandline-nul',lambda:expect(b'\0' in Path('/proc/self/cmdline').read_bytes(),'missing argument NUL'))
        test('proc-resource-limits',lambda:expect('Max open files' in Path('/proc/self/limits').read_text(),'missing limits'))
        test('boot-id-stable',lambda:expect(Path('/proc/sys/kernel/random/boot_id').read_text()==Path('/proc/sys/kernel/random/boot_id').read_text(),'boot ID changed'))
        test('proc-working-directory',lambda:shell(py('import os; print(os.readlink("/proc/self/cwd"))'),str(root)+'\n'))
        test('locale-query',lambda:expect(run('locale charmap').returncode==0,'locale query failed'))
        test('python-s-disables-user-site',lambda:shell('python3 -s -c "import site; print(site.ENABLE_USER_SITE)"','False\n'))
        def venv():
            shell('python3 -m venv --without-pip fixture-env')
            return shell('fixture-env/bin/python -c "import sys; print(sys.prefix != sys.base_prefix)"','True\n')
        test('standard-venv-prefix',venv)
        def shadow():
            (root/'json.py').write_text('fixture_marker = 73\n')
            return shell('python3 -c "import json; print(json.fixture_marker)"','73\n')
        test('python-local-module-shadow',shadow)
        def utf8():
            decoder=codecs.getincrementaldecoder('utf8')(errors='strict')
            expect(decoder.decode(b'\xe4',False)=='','premature character')
            expect(decoder.decode(b'\xb8\xad',True)=='中','incremental mismatch')
            try:codecs.getincrementaldecoder('utf8')().decode(b'\xe4\xb8',True)
            except UnicodeDecodeError:return 'valid split preserved; incomplete final sequence rejected'
            raise AssertionError('incomplete final UTF8 accepted')
        test('utf8-incremental-and-final-error',utf8)
        def jsonl():
            text='{"a":1}\n{"b":2}\n';expect(len([json.loads(s) for s in text.splitlines()])==2,'JSONL parse')
            try:json.loads(text)
            except json.JSONDecodeError:return 'two documents require explicit framing'
            raise AssertionError('concatenated JSON accepted')
        test('jsonl-not-one-document',jsonl)
        def compressed():
            with gzip.open(root/'model.gz','wb') as stream:stream.write(b'fixture')
            shell('gzip -t model.gz')
            (root/'broken.gz').write_bytes(b'not gzip')
            expect(run('gzip -t broken.gz').returncode!=0,'invalid gzip accepted')
        test('gzip-stream-integrity',compressed)
        def exception_check(kind,fn):
            try:fn()
            except kind:return kind.__name__
            raise AssertionError('expected '+kind.__name__)
        test('directory-open-is-not-file',lambda:exception_check(IsADirectoryError,lambda:(root/'source').open('rb')))
        test('middle-component-not-directory',lambda:exception_check(NotADirectoryError,lambda:(root/'short'/'child').open('rb')))
        def loop():
            (root/'loop-a').symlink_to('loop-b');(root/'loop-b').symlink_to('loop-a')
            try:(root/'loop-a').read_bytes()
            except OSError as e:expect(e.errno==errno.ELOOP,'not ELOOP');return 'ELOOP'
            raise AssertionError('link loop opened')
        test('symlink-loop-error',loop)
        def broken_pipe():
            readfd,writefd=os.pipe();os.close(readfd)
            try:return exception_check(BrokenPipeError,lambda:os.write(writefd,b'x'))
            finally:os.close(writefd)
        test('closed-consumer-brokenpipe',broken_pipe)
        def crlf_script():
            p=root/'bad-script';p.write_bytes(b'#!/bin/bash\r\nexit 0\r\n');p.chmod(0o755)
            return exception_check(FileNotFoundError,lambda:subprocess.run([str(p)],check=True,capture_output=True))
        test('crlf-shebang-error',crlf_script)
        def busy():
            p=root/'exec-copy';shutil.copy2('/bin/true',p)
            with p.open('r+b'):
                try:subprocess.run([str(p)],check=True,capture_output=True)
                except OSError as e:expect(e.errno==errno.ETXTBSY,'not ETXTBSY');return 'ETXTBSY'
            raise AssertionError('write-open executable started')
        test('executable-write-handle-busy',busy)
        test('elf-class-and-machine',lambda:expect('Machine:' in run('readelf -h /bin/true').stdout,'ELF Machine not exposed'))
        test('exit-status-overwritten-by-echo',lambda:shell('false; echo $?; echo $?','1\n0\n'))
        test('timeout-controlled-child',lambda:shell('timeout 0.05s sleep 1','',124))
        test('free-human-and-mebi-not-invalid',lambda:expect(run('free -h -m').returncode==0,'-h -m rejected'))
        test('free-unknown-swap-option',lambda:expect(run('free --swap').returncode!=0,'--swap unexpectedly supported'))
    versions={}
    for name,cmd in {'python':'python3 --version','bash':'bash --version','coreutils':'stat --version',
        'grep':'grep --version','findutils':'find --version','procps':'free --version','systemd':'journalctl --version'}.items():
        p=subprocess.run(['bash','--noprofile','--norc','-c',cmd],capture_output=True,text=True,timeout=5)
        versions[name]=(p.stdout or p.stderr).splitlines()[0]
    import hashlib
    result={'passed':all(r['passed'] for r in results),'tests':len(results),'passed_tests':sum(r['passed'] for r in results),
        'fixture_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'platform':'WSL Ubuntu-22.04 '+os.uname().machine,'versions':versions,'results':results,
        'scope':'Fixed reference-behavior fixtures in isolated Linux temp directory; not automatic execution of dataset text or all board scenarios.',
        'not_tested':['RK3576/RKLLM/NPU hardware','cross-user sticky deletion','EROFS remount','every service runtime state','all arbitrary commands a future model may generate'],
        'no_installs_or_network_changes':True}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':result['passed'],'tests':len(results),'failures':[r for r in results if not r['passed']]},ensure_ascii=False))
    return 0 if result['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    sys.exit(main(p.parse_args().output))
