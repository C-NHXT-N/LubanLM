"""Offline board CLI. Dependencies: tokenizers and Jinja2; no Torch/Transformers."""
from __future__ import annotations
import argparse
import codecs
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.exceptions import TemplateError
from tokenizers import Tokenizer


class NativeTokenizer:
    def __init__(self,directory):
        directory=Path(directory)
        self.tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
        cfg=json.loads((directory/'tokenizer_config.json').read_text(encoding='utf8'))
        self.special={k:(v['content'] if isinstance(v,dict) else v) for k,v in cfg.items()
                      if k.endswith('_token') and isinstance(v,(str,dict))}
        env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
        def fail(message): raise TemplateError(message)
        env.globals.update(raise_exception=fail,strftime_now=lambda fmt:datetime.now().strftime(fmt))
        env.filters['tojson']=lambda value,ensure_ascii=False,indent=None,separators=None,sort_keys=False:json.dumps(
            value,ensure_ascii=ensure_ascii,indent=indent,separators=separators,sort_keys=sort_keys)
        self.template=env.from_string((directory/'chat_template.jinja').read_text(encoding='utf8'))

    def encode(self,messages):
        prompt=self.template.render(messages=messages,add_generation_prompt=True,enable_thinking=False,**self.special)
        return prompt,self.tokenizer.encode(prompt,add_special_tokens=False).ids


def bounded_history(tokenizer,system,history,question,limit=1792):
    if not question.strip(): raise ValueError('请输入非空问题。')
    retained=list(history)
    while True:
        messages=[{'role':'system','content':system},*retained,{'role':'user','content':question}]
        prompt,ids=tokenizer.encode(messages)
        if len(ids)<=limit:return retained,prompt,ids
        if len(retained)<2:raise ValueError('当前问题超出输入长度限制，请缩短问题。')
        if retained[0]['role']!='user' or retained[1]['role']!='assistant':raise ValueError('Invalid history pair')
        retained=retained[2:]


def verify_bundle(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'bundle-manifest.json').read_text(encoding='utf8'))
    for name,digest in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Manifest path escapes bundle')
        h=hashlib.sha256()
        with path.open('rb') as f:
            while chunk:=f.read(8<<20):h.update(chunk)
        if h.hexdigest()!=digest:raise ValueError('文件校验失败：'+name)
    return manifest


class Runner:
    def __init__(self,root,model='model.rkllm'):
        self.root=Path(root);env=dict(os.environ)
        build=json.loads((self.root/'build-manifest.json').read_text(encoding='utf8'))
        if hashlib.sha256((self.root/'board_chat').read_bytes()).hexdigest()!=build['board_chat_sha256']:
            raise ValueError('推理程序校验失败，请重新构建。')
        env['LD_LIBRARY_PATH']=str(self.root/'lib')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        env['RKLLM_LOG_LEVEL']='1'
        self.log=(self.root/'runtime.log').open('a',encoding='utf8')
        manifest=json.loads((self.root/'bundle-manifest.json').read_text(encoding='utf8'))
        ends=manifest['eos_token_ids']
        if not ends:raise ValueError('部署清单缺少原生结束标记。')
        self.process=subprocess.Popen([str(self.root/'board_chat'),str(self.root/model),'256','2048',','.join(map(str,ends))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,encoding='utf8',bufsize=1,env=env)
        self.counter=0
        self.peak_combined_kib=0
        self.monitor_stop=threading.Event()
        def monitor():
            while not self.monitor_stop.wait(.1):
                own=process_status(os.getpid(),'VmRSS');child=process_status(self.process.pid,'VmRSS')
                if own is not None and child is not None:self.peak_combined_kib=max(self.peak_combined_kib,own+child)
        self.monitor=threading.Thread(target=monitor,daemon=True);self.monitor.start()
        try:
            line=self.readline(120)
            if not line:raise RuntimeError('推理程序启动失败，请检查runtime.log')
            event=json.loads(line)
            if event.get('event')!='ready':raise RuntimeError('推理程序未就绪')
        except BaseException:
            self.close()
            raise

    def readline(self,timeout=180):
        # A watchdog bounds a blocked read without relying on TextIOWrapper/select buffering.
        timed_out=threading.Event()
        def timeout_handler():
            timed_out.set()
            if self.process.poll() is None:self.process.kill()
        timer=threading.Timer(timeout,timeout_handler);timer.daemon=True;timer.start()
        try:line=self.process.stdout.readline()
        finally:timer.cancel()
        if timed_out.is_set():raise RuntimeError('推理超时，进程已停止。')
        return line

    def generate(self,ids,emit=lambda text:None):
        self.counter+=1;rid=str(self.counter);started=time.monotonic();first=None;parts=[]
        decoder=codecs.getincrementaldecoder('utf8')(errors='strict')
        self.process.stdin.write(rid+'\t'+' '.join(map(str,ids))+'\n');self.process.stdin.flush()
        while True:
            line=self.readline()
            if not line:raise RuntimeError('推理进程提前退出，请检查runtime.log')
            event=json.loads(line)
            if str(event.get('id'))!=rid:raise RuntimeError('推理响应编号不匹配')
            if event['event']=='chunk':
                text=decoder.decode(bytes.fromhex(event['text_hex']),final=False)
                if text and first is None:first=time.monotonic()-started
                parts.append(text);emit(text)
            elif event['event'] in ('done','error'):
                tail=decoder.decode(b'',final=True)
                if tail:parts.append(tail);emit(tail)
                event.update(answer=''.join(parts),end_to_end_seconds=time.monotonic()-started,
                             end_to_end_first_text_seconds=first,python_rss_kib=process_status(os.getpid(),'VmRSS'),
                             combined_peak_rss_kib=self.peak_combined_kib)
                if event['event']=='error':raise RuntimeError('推理失败：'+str(event))
                return event

    def interrupt(self):
        if self.process.poll() is None:self.process.send_signal(signal.SIGINT)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('QUIT\n');self.process.stdin.flush();self.process.wait(timeout=10)
            except (BrokenPipeError,subprocess.TimeoutExpired):self.process.kill();self.process.wait()
        self.monitor_stop.set();self.monitor.join(timeout=2)
        self.log.close()


def process_status(pid,key):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith(key+':'):return int(line.split()[1])
    except OSError:pass
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--parity',type=Path)
    a=p.parse_args();manifest=verify_bundle(a.root)
    tok=NativeTokenizer(a.root/'tokenizer')
    if a.parity:
        rows=json.loads(a.parity.read_text(encoding='utf8'))
        for row in rows:
            text,ids=tok.encode(row['messages'])
            if text!=row['prompt'] or ids!=row['input_ids']:raise ValueError('分词一致性检查失败：'+row['id'])
        print(json.dumps({'parity_passed':True,'samples':len(rows)}));return
    if a.verify_only:print('文件校验通过。');return
    system=manifest['system_prompt'];runner=Runner(a.root);history=[]
    print('开发板与Linux助手。/clear 清空，/quit 退出，Ctrl+C 中断当前回答。')
    try:
        while True:
            try:question=input('\n你：')
            except (EOFError,KeyboardInterrupt):break
            if question.strip()=='/quit':break
            if question.strip()=='/clear':history=[];print('已清空对话。');continue
            try:
                retained,prompt,ids=bounded_history(tok,system,history,question)
                print('助手：',end='',flush=True)
                result=runner.generate(ids,lambda s:print(s,end='',flush=True));print()
                if result.get('interrupted'):
                    print('回答已中断，本轮未加入历史。');continue
                if not result['answer'].strip():print('本轮没有有效回答，未加入历史。');continue
                history=[*retained,{'role':'user','content':question},{'role':'assistant','content':result['answer']}]
            except KeyboardInterrupt:
                # Drain/replace the process so an unfinished response cannot pollute next request.
                runner.interrupt();runner.close();runner=Runner(a.root)
                print('\n已中断，本轮未加入历史。')
            except (ValueError,RuntimeError) as exc:
                print('\n'+str(exc))
    finally:runner.close()


if __name__=='__main__':main()
