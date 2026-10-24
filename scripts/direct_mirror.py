"""Verified HTTPS on a physical Windows interface; never use proxies/TUN DNS."""
from __future__ import annotations
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import socket
import ssl
import struct
import subprocess
import threading
import time
from urllib.parse import urljoin, urlsplit, quote

class PhysicalMirror:
    endpoint = 'https://hf-mirror.com'

    def __init__(self, audit=None, interface=None, bind_ip=None, allowed_hosts=None):
        if os.name != 'nt':
            raise RuntimeError('Physical-interface downloader requires Windows')
        interface = interface or os.environ.get('LUBAN_DOWNLOAD_INTERFACE')
        bind_ip = bind_ip or os.environ.get('LUBAN_DOWNLOAD_BIND_IP')
        if bind_ip:
            bind_ip = str(ipaddress.IPv4Address(bind_ip))
        command = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); @(Get-NetAdapter -Physical | Where-Object Status -eq 'Up' | ForEach-Object { $a=$_; $c=Get-NetIPConfiguration -InterfaceIndex $a.ifIndex; if($c.IPv4DefaultGateway){[ordered]@{index=$a.ifIndex;name=$a.Name;ip=@($c.IPv4Address.IPAddress)[0];description=$a.InterfaceDescription}} }) | ConvertTo-Json -Compress"
        value = json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',command], encoding='utf-8', errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
        rows = value if isinstance(value,list) else ([value] if value else [])
        if interface:
            rows = [r for r in rows if r['name'] == interface]
        if bind_ip:
            rows = [r for r in rows if r['ip'] == bind_ip]
        if not rows:
            raise RuntimeError(f'No physical IPv4 adapter with gateway matches interface={interface!r}, bind_ip={bind_ip!r}; VPN fallback forbidden')
        rows.sort(key=lambda r: r['index'])
        self.adapter, self.audit = rows[0], Path(audit) if audit else None
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts) if allowed_hosts is not None else None
        ipaddress.IPv4Address(self.adapter['ip'])
        self.lock, self.cache, self.good_ip = threading.RLock(), {}, {}
        self.record(event='physical_adapter', requested_interface=interface, requested_bind_ip=bind_ip, **self.adapter)

    def record(self, **row):
        if self.audit:
            with self.lock:
                self.audit.parent.mkdir(parents=True,exist_ok=True)
                with self.audit.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(timestamp=time.time(),**row),ensure_ascii=False)+'\n')

    def check_host(self, hostname):
        if self.allowed_hosts is not None and hostname.lower() not in self.allowed_hosts:
            self.record(event='host_rejected', host=hostname)
            raise ValueError(f'Download host not approved: {hostname}')

    def socket_connection(self, hostname, addresses):
        """Raw TCP for a TLS client or a CONNECT tunnel; never use the default route."""
        self.check_host(hostname)
        last = None
        ordered = sorted(addresses,key=lambda ip: ip != self.good_ip.get(hostname))
        for address in ordered:
            ipaddress.IPv4Address(address)
            sock = None
            try:
                sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.setsockopt(socket.IPPROTO_IP,31,struct.pack('!I',int(self.adapter['index'])))
                if sock.getsockopt(socket.IPPROTO_IP,31)!=int(self.adapter['index']):
                    raise RuntimeError('Physical interface socket option did not persist')
                sock.bind((self.adapter['ip'],0))
                sock.connect((address,443))
                if sock.getsockname()[0] != self.adapter['ip']:
                    raise RuntimeError('Physical binding verification failed')
                self.good_ip[hostname] = address
                self.record(event='physical_tcp_connection',host=hostname,remote_ip=address,local_ip=sock.getsockname()[0],interface_index=self.adapter['index'])
                return sock
            except OSError as exc:
                last = exc
                if sock is not None: sock.close()
            except BaseException:
                if sock is not None: sock.close()
                raise
        raise ConnectionError(f'No reachable physical HTTPS address for {hostname}: {last}')

    def connection(self, hostname, addresses):
        connection = http.client.HTTPSConnection(hostname,timeout=45,context=ssl.create_default_context())
        sock = self.socket_connection(hostname, addresses)
        try:
            connection.sock = connection._context.wrap_socket(sock,server_hostname=hostname)
            connection.sock.settimeout(45)
            self.record(event='https_connection',host=hostname,remote_ip=connection.sock.getpeername()[0],local_ip=connection.sock.getsockname()[0],interface_index=self.adapter['index'])
            return connection
        except BaseException:
            sock.close()
            connection.close()
            raise

    def resolve(self, host):
        self.check_host(host)
        with self.lock:
            cached = self.cache.get(host)
            if cached and cached[0] > time.time(): return cached[1]
        connection = self.connection('dns.alidns.com',['223.5.5.5','223.6.6.6'])
        try:
            connection.request('GET','/resolve?name='+quote(host)+'&type=A',headers={'Accept':'application/dns-json'})
            response = connection.getresponse()
            if response.status != 200: raise ConnectionError('Encrypted DNS failed')
            value = json.loads(response.read(1<<20))
            ips = [r['data'] for r in value.get('Answer',[]) if r.get('type') == 1]
            if not ips: raise ConnectionError(f'No real IPv4 DNS answers for {host}')
            for ip in ips:
                if not ipaddress.ip_address(ip).is_global:
                    raise ConnectionError('DNS supplied a private/FakeIP address')
            with self.lock: self.cache[host] = (time.time()+300,ips)
            return ips
        finally: connection.close()

    def open(self,url,headers=None):
        for _ in range(8):
            parts = urlsplit(url)
            if parts.scheme != 'https' or parts.username or parts.port not in (None,443):
                raise ValueError('Only verified HTTPS download URLs allowed')
            connection = self.connection(parts.hostname,self.resolve(parts.hostname))
            try:
                path = parts.path + ('?'+parts.query if parts.query else '')
                connection.request('GET',path,headers={'User-Agent':'LubanLM-local-research/10b','Accept-Encoding':'identity',**(headers or {})})
                response = connection.getresponse()
                if response.status in (301,302,303,307,308):
                    location = response.getheader('Location')
                    connection.close()
                    if not location: raise ConnectionError('Redirect without Location')
                    url = urljoin(url,location)
                    continue
                return connection,response
            except BaseException:
                connection.close()
                raise
        raise ConnectionError('Too many download redirects')

    def bytes(self,url,maximum=20<<20,headers=None):
        connection,response = self.open(url,headers)
        try:
            if response.status not in (200,206): raise ConnectionError(f'HTTP {response.status}')
            result=response.read(maximum+1)
            if len(result)>maximum: raise ValueError('Response exceeds bounded size')
            return result
        finally: connection.close()

    def metadata(self,repo,revision='main'):
        url=f'{self.endpoint}/api/datasets/{repo}/revision/{revision}?blobs=true'
        result=json.loads(self.bytes(url))
        if revision!='main' and result['sha']!=revision: raise ValueError('Revision mismatch')
        return result

    def download(self,url,path,size,digest,guard=lambda:None):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        def hashed(p):
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1<<20),b''): h.update(chunk)
            return h.hexdigest()
        if path.exists():
            if path.stat().st_size==size and hashed(path)==digest: return path
            raise ValueError(f'Existing file checksum mismatch: {path}')
        part=path.with_suffix(path.suffix+'.part')
        for attempt in range(6):
            guard()
            offset=part.stat().st_size if part.exists() else 0
            if offset>size: raise ValueError('Partial file exceeds expected length')
            if offset==size:
                if hashed(part)!=digest:
                    part.unlink(); raise ValueError('Downloaded SHA256 mismatch')
                part.replace(path); return path
            connection=None
            try:
                connection,response=self.open(url,{'Range':f'bytes={offset}-'} if offset else {})
                if response.status not in (200,206): raise ConnectionError(f'HTTP {response.status}')
                if response.status==206 and not response.getheader('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Invalid resume Content-Range')
                if offset and response.status==200: offset=0
                with part.open('ab' if offset else 'wb') as stream:
                    while chunk:=response.read(1<<20):
                        guard(); offset+=len(chunk)
                        if offset>size: raise ValueError('Download exceeds metadata size')
                        stream.write(chunk)
                if offset!=size: raise ConnectionError('Incomplete download')
                if hashed(part)!=digest:
                    part.unlink(); raise ValueError('Downloaded SHA256 mismatch')
                part.replace(path); return path
            except (OSError,http.client.HTTPException) as exc:
                self.record(event='download_retry',attempt=attempt+1,error=type(exc).__name__,file=path.name)
                if attempt==5: raise
                time.sleep(min(30,2**attempt))
            finally:
                if connection: connection.close()
        raise RuntimeError('Download exhausted retries')
