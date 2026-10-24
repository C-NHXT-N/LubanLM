"""Quota allocation by maximum flow; avoids source-order quota starvation."""
from collections import Counter,deque,defaultdict

def soft_allocations(counts,source_caps,total,daily_min,concept_max):
    """Reserve the daily lower bound, then augment without losing that bound."""
    repos=sorted({repo for task,repo in counts})
    present={task for task,repo in counts}
    # Rare real tasks get first access to the flexible half of the budget;
    # abundant daily/concept rows must not silently crowd them out.
    preferred=['greeting','identity','arithmetic','concept','instruction','daily']
    tasks=[t for t in preferred if t in present]+sorted(present-set(preferred))
    graph=defaultdict(dict)
    def edge(a,b,n):graph[a][b]=n;graph[b].setdefault(a,0)
    for task in tasks:
        edge('start','task:'+task,daily_min if task=='daily' else 0)
        for repo in repos:edge('task:'+task,'repo:'+repo,counts.get((task,repo),0))
    for repo in repos:edge('repo:'+repo,'end',int(source_caps.get(repo,1)*total))
    def augment(limit):
        filled=0
        while filled<limit:
            parent={'start':None};queue=deque(['start'])
            while queue and 'end' not in parent:
                node=queue.popleft()
                for other,capacity in graph[node].items():
                    if capacity>0 and other not in parent:parent[other]=node;queue.append(other)
            if 'end' not in parent:break
            take=limit-filled;node='end'
            while parent[node] is not None:take=min(take,graph[parent[node]][node]);node=parent[node]
            node='end'
            while parent[node] is not None:
                prev=parent[node];graph[prev][node]-=take;graph[node][prev]+=take;node=prev
            filled+=take
        return filled
    minimum=augment(daily_min)
    if minimum<daily_min:return {},dict(daily_gap=daily_min-minimum)
    for task in tasks:
        cap=concept_max if task=='concept' else total
        graph['start']['task:'+task]+=cap-(daily_min if task=='daily' else 0)
    filled=minimum+augment(total-minimum)
    allocation={(t,r):graph['repo:'+r]['task:'+t] for t in tasks for r in repos}
    return allocation,dict(total_gap=total-filled,daily_gap=0)

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
