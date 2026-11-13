"""One-time audited copy of an unchanged legacy selection plus new sources.

Fail closed on old rule/config changes; never change the preserved old database.
"""
import ast,json,sqlite3,shutil,argparse
from pathlib import Path
from scripts.common import ROOT,read_json,write_json,sha256,utc
from scripts.policy_versions import sft_identity

BACKUP=Path('D:/LubanLM-Runs/general-10b-teaching-v2/recovery/supplement-20261007')

def function(path,name):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name==name)

def compatible_parser(old,new):
    for name in ('category',):
        assert ast.dump(function(old,name))==ast.dump(function(new,name)),name
    a,b=function(old,'messages'),function(new,'messages')
    b.body=[n for n in b.body if not (isinstance(n,ast.If) and ("'dialog'" in ast.unparse(n.test) or "'conversation'" in ast.unparse(n.test)))]
    assert ast.dump(a)==ast.dump(b),'legacy messages adapter changed'
    a,b=function(old,'candidates'),function(new,'candidates')
    allowed={'allowed_tasks','daily_only','human_dialogue','exclude_context_regex','first_daily_only'}
    class RemoveAdditions(ast.NodeTransformer):
        def visit_If(self,node):
            expression=ast.unparse(node.test)
            if "source['repo'] == 'allenai/WildChat-1M'"==expression:return None
            if any("source.get('"+key+"')" in expression for key in allowed):return None
            return self.generic_visit(node)
    b=RemoveAdditions().visit(b)
    assert ast.dump(a)==ast.dump(b),'legacy candidate rules changed'
    a,b=function(old,'rows'),function(new,'rows')
    # The new first branch is gzip; its else is exactly the old reader.
    branch=b.body[0]
    while isinstance(branch,ast.If) and ("'.jsonl.gz'" in ast.unparse(branch.test) or "'.zip'" in ast.unparse(branch.test)):
        branch=branch.orelse[0]
    assert ast.dump(a.body[0])==ast.dump(branch),'legacy raw reader changed'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adopt-existing',action='store_true');args=parser.parse_args()
    plan_path=ROOT/'configs/50m-general-pretrain-10b-teaching.json';plan=read_json(plan_path)
    old_plan=read_json(BACKUP/'configs'/plan_path.name)
    a=dict(old_plan['sft']);b=dict(plan['sft'])
    for v in (a,b):
        for k in ('data_version','raw_sources','skip_legacy_archive'):v.pop(k,None)
    assert a==b and old_plan['seed']==plan['seed'] and old_plan['tokenizer_sha256']==plan['tokenizer_sha256'],'selection settings changed'
    original=old_plan['sft']['raw_sources']
    assert plan['sft']['raw_sources'][:len(original)]==original,'original pinned sources changed'
    compatible_parser(BACKUP/'scripts/prepare_sft_100k.py',ROOT/'scripts/prepare_sft_100k.py')
    for name in ('insert_candidate','collapse_lineage','assign_splits','finalize'):
        assert ast.dump(function(BACKUP/'scripts/sft_disk.py',name))==ast.dump(function(ROOT/'scripts/sft_disk.py',name)),name+' changed'
    src=Path(plan['data_root'])/'processed'/old_plan['sft']['data_version']/'candidates.sqlite'
    dst=src.parent.parent/plan['sft']['data_version']/'candidates.sqlite'
    if dst.exists() and not args.adopt_existing:raise FileExistsError(dst)
    old=sqlite3.connect('file:'+src.as_posix()+'?mode=ro',uri=True)
    try:
        policy=json.loads(old.execute("SELECT value FROM meta WHERE name='policy'").fetchone()[0]);current=sft_identity(plan_path)
        for k in ('blocked_sha256','frozen_sha256'):assert policy[k]==current[k],k+' changed'
        for f,h in policy['code'].items():
            expected=sha256(BACKUP/f) if f in ('scripts/sft_disk.py','scripts/prepare_sft_100k.py') else sha256(ROOT/f)
            assert h==expected,f+' changed'
        assert not old.execute('SELECT 1 FROM cursors WHERE complete=0').fetchone(),'old scan incomplete'
        n=old.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
        if shutil.disk_usage(dst.parent.parent).free-src.stat().st_size<plan['disk_free_bytes_floor']:raise RuntimeError('RESOURCE_WAIT: copy reserve')
        dst.parent.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(dst,uri=True)
        try:
            if args.adopt_existing:
                # This special recovery is allowed only before any supplement
                # was accepted: compare every immutable old candidate payload.
                db.execute('ATTACH DATABASE ? AS previous',('file:'+src.as_posix()+'?mode=ro',))
                assert db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]==n,'supplement already accepted'
                fields=('context','exact','task','source','rank','payload')
                conditions=' OR '.join('a.'+k+'<>b.'+k for k in fields)
                assert not db.execute('SELECT 1 FROM candidates a LEFT JOIN previous.candidates b ON a.id=b.id WHERE b.id IS NULL OR '+conditions+' LIMIT 1').fetchone(),'candidate contents changed'
                assert not db.execute('SELECT 1 FROM cursors a LEFT JOIN previous.cursors b ON a.file=b.file WHERE b.file IS NULL OR a.position<>b.position OR a.complete<>b.complete OR a.counts<>b.counts LIMIT 1').fetchone(),'new cursor work present'
                db.execute('DETACH DATABASE previous')
            else:old.backup(db,pages=128)
            assert db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]==n,'copy count mismatch'
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('policy',json.dumps(current)))
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('compatible_reuse',json.dumps(dict(previous_policy=policy,previous_database=str(src),created_utc=utc()))));db.commit()
        finally:db.close()
    finally:old.close()
    write_json(Path(plan['runs_root'])/'reports/sft_compatible_reuse.json',dict(created_utc=utc(),previous_database=str(src),new_database=str(dst),candidates=n,legacy_rules_ast_equal=True,settings_and_original_sources_equal=True,remaining_action='Process supplementary sources, reassign splits and review 600 selected conversations',previous_policy=policy,new_policy=current))
    print('Adopted unchanged candidates:' if args.adopt_existing else 'Copied unchanged candidates:',n)

if __name__=='__main__':main()
