"""Board-only persistent Tailscale DIRECT rules, validated before local API reload.

No proxy subscription or credential is printed or copied off the board.
"""
from pathlib import Path
import json
import shutil
import subprocess
import urllib.request
import yaml

RULES=['PROCESS-NAME,tailscaled,DIRECT','PROCESS-NAME,tailscale,DIRECT',
       'IP-CIDR,100.64.0.0/10,DIRECT,no-resolve','IP-CIDR6,fd7a:115c:a1e0::/48,DIRECT,no-resolve',
       'DOMAIN-SUFFIX,tailscale.com,DIRECT','DOMAIN-SUFFIX,tailscale.io,DIRECT','DOMAIN-SUFFIX,ts.net,DIRECT']

def main():
    root=Path('/home/cat/clashctl/resources');binary='/home/cat/clashctl/bin/mihomo'
    mixin=root/'mixin.yaml';runtime=root/'runtime.yaml'
    previous={p:p.read_bytes() for p in [mixin,runtime]}
    original=yaml.safe_load(previous[runtime]);data=yaml.safe_load(previous[mixin])
    if str(original.get('mode','rule')).lower()!='rule':raise RuntimeError('Expected Rule mode; refusing broader mode change')
    rules=data.setdefault('rules',{})
    if not isinstance(rules,dict):raise RuntimeError('Unexpected clashctl mixin rules schema')
    rules['prefix']=RULES+[r for r in rules.get('prefix') or [] if r not in RULES]
    new_mixin=yaml.safe_dump(data,allow_unicode=True,sort_keys=False).encode()
    live=dict(original);live['rules']=RULES+[r for r in original.get('rules',[]) if r not in RULES]
    candidate=root/'runtime.tailscale-candidate.yaml'
    candidate.write_text(yaml.safe_dump(live,allow_unicode=True,sort_keys=False),encoding='utf8')
    check=subprocess.run([binary,'-t','-d',str(root),'-f',str(candidate)],capture_output=True)
    if check.returncode!=0:
        candidate.unlink();raise RuntimeError('Mihomo rejected candidate; existing configuration untouched')
    headers={'Content-Type':'application/json'}
    if original.get('secret'):headers['Authorization']='Bearer '+original['secret']
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def reload_config():
        request=urllib.request.Request('http://127.0.0.1:9090/configs',
            data=json.dumps({'path':str(runtime)}).encode(),headers=headers,method='PUT')
        with opener.open(request,timeout=20) as response:
            if response.status!=204:raise RuntimeError('Unexpected reload response')
    try:
        for p in [mixin,runtime]:
            backup=p.with_name(p.name+'.before-tailscale-20261018')
            if not backup.exists():shutil.copy2(p,backup)
        mixin.write_bytes(new_mixin);candidate.replace(runtime)
        reload_config()
        request=urllib.request.Request('http://127.0.0.1:9090/rules',headers=headers)
        with opener.open(request,timeout=15) as response:active=json.load(response)['rules']
        if active[0].get('payload')!='tailscaled':raise RuntimeError('Live rule verification failed')
    except BaseException:
        for p,content in previous.items():p.write_bytes(content)
        reload_config();raise
    report={'completed':True,'persistent_file':str(mixin),'runtime_file':str(runtime),
            'backup_suffix':'.before-tailscale-20261018','rules':RULES,
            'mode':live.get('mode'),'tun_enabled':live.get('tun',{}).get('enable'),
            'verified_live_rule_prefix':active[:len(RULES)],'hot_reload':True}
    result=Path('/home/cat/lubanlm-qwen-20261018/results/tailscale-direct.json')
    result.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
