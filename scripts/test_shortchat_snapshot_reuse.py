"""Verify copy/update backend preserves actual candidate content and dedup index."""
import importlib.util,json,sqlite3,unittest
from pathlib import Path

class SnapshotTests(unittest.TestCase):
    def test_unchanged_changed_and_new_candidates(self):
        spec=importlib.util.spec_from_file_location('snapshot_backend',Path('.cache/fast-reuse-shortchat-v3.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE candidates(id TEXT PRIMARY KEY,original TEXT,qhash TEXT,exact TEXT,family TEXT,split TEXT,category TEXT,source TEXT,short INT,payload TEXT,rank TEXT,answer_cost INT)');db.execute('CREATE TABLE near(bucket TEXT,fp TEXT)')
        sql='INSERT OR IGNORE INTO candidates(id,original,qhash,exact,family,split,category,source,short,payload,rank,answer_cost) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)'
        args=('same','original','qhash','exact','family','train','boundary','source',1,json.dumps({'category':'boundary','labels':[2]}),'rank',-2)
        db.execute(sql,args);db.execute('INSERT INTO near VALUES(?,?)',('bucket','fp'))
        proxy=module.SnapshotConnection(db,{'same':'boundary'})
        before=db.total_changes;proxy.execute(sql,args);self.assertEqual(db.total_changes,before)
        changed=list(args);changed[6]='instruction';changed[9]=json.dumps({'category':'instruction','labels':[2]});proxy.execute(sql,changed)
        row=db.execute('SELECT original,qhash,exact,family,category,payload FROM candidates').fetchone()
        self.assertEqual(row[:4],args[1:5]);self.assertEqual(row[4],'instruction');self.assertEqual(json.loads(row[5])['labels'],[2])
        new=list(args);new[0]='new';proxy.execute(sql,new);self.assertEqual(db.execute('SELECT count(*) FROM candidates').fetchone()[0],2)
        proxy.execute('DELETE FROM near');proxy.executemany('INSERT INTO near VALUES(?,?)',[('bucket','fp')]);self.assertEqual(db.execute('SELECT count(*) FROM near').fetchone()[0],1)
        db.close()

if __name__=='__main__':unittest.main()
