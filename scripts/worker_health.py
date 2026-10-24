"""Small process leases and CPU accounting; no training/data policy changes."""
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
import psutil
from scripts.common import read_json, write_json, utc
from scripts.policy_versions import digest

def load(path, default=None):
    try:return read_json(path)
    except (FileNotFoundError, ValueError):return {} if default is None else default

def process_matches(pid, created=None, modules=()):
    try:
        p=psutil.Process(int(pid))
        return (created is None or abs(p.create_time()-float(created))<.1) and p.is_running() and (not modules or any(m in p.cmdline() for m in modules))
    except (psutil.Error, ValueError, TypeError):return False

def acquire_lock(path, modules):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    owner=dict(version=2,pid=os.getpid(),created=psutil.Process().create_time(),token=uuid.uuid4().hex,modules=list(modules))
    for _ in range(2):
        try:
            with path.open('x',encoding='utf-8') as f:
                json.dump(owner,f);f.flush();os.fsync(f.fileno())
            return owner
        except FileExistsError:
            text=path.read_text(encoding='utf-8').strip()
            try:
                saved=json.loads(text)
                if isinstance(saved,int):saved=dict(pid=saved)
            except ValueError:saved={}
            if isinstance(saved,dict) and process_matches(saved.get('pid'),saved.get('created'),() if saved.get('created') is not None else modules):return None
            # An empty lock may still be in the tiny exclusive-create/write window.
            if not text and time.time()-path.stat().st_mtime<5:return None
            # Compare before deletion; never remove a replacement owner's lock.
            if path.read_text(encoding='utf-8').strip()==text:path.unlink(missing_ok=True)
    return None

def release_lock(path,owner):
    if owner and load(path).get('token')==owner['token']:Path(path).unlink(missing_ok=True)

def durable_progress(data):
    """Read committed SQLite state, including its WAL, never in-flight reports."""
    data=Path(data);result={}
    for path,kind in [(data/'dedup.sqlite','pretrain'),*[(p,'sft') for p in (data/'processed').glob('sft*/candidates.sqlite')]]:
        if not path.exists():continue
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=2)
        try:
            db.execute('PRAGMA query_only=ON')
            if kind=='pretrain':
                for name,text in db.execute('SELECT name,state FROM checkpoints'):
                    row=json.loads(text);result[name]=dict(tokens=row.get('train',{}).get('tokens',0),validation=row.get('validation',{}).get('tokens',0),files=len(row.get('done_files',[])),rows=row.get('file_rows',row.get('cursors',{})))
            else:
                result[str(path.relative_to(data))]=list(db.execute('SELECT file,position,complete FROM cursors ORDER BY file'))
        finally:db.close()
    return result

def work_signature(plan,pid):
    p=psutil.Process(pid);times=p.cpu_times()
    # CPU activity permits long hashing/index-building operations without false stalls.
    value=dict(cpu_bucket=math.floor((times.user+times.system)/5))
    reports=Path(plan['runs_root'])/'reports'
    for name in ('preparation_progress.json','sft_preparation_progress.json'):
        row=load(reports/name)
        value[name]={k:row[k] for k in ('source','file','position','train_tokens','files') if k in row}
        value[name]['raw_documents']=row.get('counts',{}).get('raw_documents')
    value['partials']=[]
    for root in {Path(plan['data_root'])/'raw-cache',Path(plan.get('sft_raw_root',Path(plan['data_root'])/'sft-raw'))}:
        for f in root.rglob('*.part'):
            try:value['partials'].append((str(f),f.stat().st_size))
            except FileNotFoundError:pass
    value['metrics']=[]
    for f in (Path(plan['runs_root'])/'checkpoints').glob('*/metrics.jsonl'):
        try:value['metrics'].append((str(f),f.stat().st_size))
        except FileNotFoundError:pass
    value['partials'].sort();value['metrics'].sort()
    return digest(value)

def health_check(previous,signature,heartbeat_age,now=None):
    now=time.time() if now is None else now
    changed=previous.get('work_signature')!=signature
    since=now if changed else previous.get('work_changed_at',now)
    suspect=heartbeat_age>180 or now-since>=1800
    confirmed=suspect and previous.get('stall_suspected_at') is not None and (heartbeat_age>180 or not changed)
    return dict(work_signature=signature,work_changed_at=since,stall_suspected_at=(previous.get('stall_suspected_at') or now) if suspect else None),confirmed

def reconcile_cpu(path,pid,created,phase,started_at,finished=False,ended_at=None):
    """Absolute execution interval makes repeated reconciliation idempotent."""
    lock=Path(path).with_suffix('.lease.lock');owner=None
    for _ in range(30):
        owner=acquire_lock(lock,('scripts.prepare_general_10b','scripts.prepare_sft_100k','scripts.run_general_10b','scripts.teaching_supervisor'))
        if owner is not None:break
        time.sleep(.1)
    if owner is None:raise RuntimeError('CPU budget accounting lock busy')
    try:
        entries=load(path,[]);entry=next((r for r in entries if r.get('pid')==pid and r.get('created')==created),None)
        if entry is None:
            entry=dict(version=2,pid=pid,created=created,phase=phase,started_at=started_at,started_utc=datetime.fromtimestamp(started_at,timezone.utc).isoformat(),seconds=0,finished=False);entries.append(entry)
        if not entry.get('finished'):
            end=time.time() if ended_at is None else ended_at
            entry.update(seconds=max(entry['seconds'],max(0,end-entry['started_at'])),updated_at=end,finished=finished)
        write_json(path,entries);return sum(r['seconds'] for r in entries)
    finally:release_lock(lock,owner)

class WorkerHealth:
    def __init__(self,plan_path,phase):
        self.plan=read_json(plan_path);self.phase=phase;self.started=time.time();self.created=psutil.Process().create_time();self.stop=threading.Event()
        self.report=Path(self.plan['runs_root'])/'reports/worker_heartbeat.json'
        self.budget=Path(self.plan['runs_root'])/'reports/cpu_preparation_budget.json'
    def pulse(self,finished=False):
        spent=reconcile_cpu(self.budget,os.getpid(),self.created,self.phase,self.started,finished)
        write_json(self.report,dict(version=2,pid=os.getpid(),created=self.created,phase=self.phase,updated_utc=utc(),work_signature=work_signature(self.plan,os.getpid()),cpu_seconds_total=spent,finished=finished))
        if spent>=self.plan['preparation_hours_cap']*3600 and not finished:raise RuntimeError('PREPARATION_BUDGET_EXHAUSTED')
    def loop(self):
        while not self.stop.wait(10):
            try:self.pulse()
            except BaseException as exc:
                import traceback
                traceback.print_exception(type(exc),exc,exc.__traceback__)
                os._exit(70) # Fail closed rather than silently losing budget enforcement.
    def __enter__(self):
        self.pulse();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.pulse(finished=True)
