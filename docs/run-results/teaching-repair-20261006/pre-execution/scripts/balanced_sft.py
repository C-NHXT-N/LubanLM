"""Quota allocation by maximum flow; avoids source-order quota starvation."""
from collections import Counter,deque,defaultdict

def select_balanced(pool,targets,source_caps,total):
    repos=sorted({r['source'] for r in pool});tasks=list(targets)
    counts=Counter((r['task'],r['source']) for r in pool)
    graph=defaultdict(dict)
    def edge(a,b,n):graph[a][b]=n;graph[b].setdefault(a,0)
    for task,n in targets.items():
        edge('start','task:'+task,n)
        for repo in repos:edge('task:'+task,'repo:'+repo,counts[task,repo])
    for repo in repos:edge('repo:'+repo,'end',int(source_caps.get(repo,1)*total))
    while True:
        parent={'start':None};queue=deque(['start'])
        while queue and 'end' not in parent:
            node=queue.popleft()
            for other,capacity in graph[node].items():
                if capacity>0 and other not in parent:parent[other]=node;queue.append(other)
        if 'end' not in parent:break
        take=total;node='end'
        while parent[node] is not None:take=min(take,graph[parent[node]][node]);node=parent[node]
        node='end'
        while parent[node] is not None:
            prev=parent[node];graph[prev][node]-=take;graph[node][prev]+=take;node=prev
    allocations={(task,repo):graph['repo:'+repo]['task:'+task] for task in tasks for repo in repos}
    chosen=[]
    for row in pool:
        pair=row['task'],row['source']
        if allocations.get(pair,0)>0:chosen.append(row);allocations[pair]-=1
    return chosen
