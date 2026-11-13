"""Integral minimum-cost allocation under exact category and source caps."""
from collections import defaultdict,deque
from scripts.shortchat_common import CATEGORIES

def source_cap(source,total):
    return total//10 if source=='rule-basics-v1' else int(total*(.15 if source=='BelleGroup/train_1M_CN' else .4))

def add_edge(edges,a,b,capacity,cost):
    forward=[b,capacity,cost,len(edges[b])];back=[a,0,-cost,len(edges[a])]
    edges[a].append(forward);edges[b].append(back);return forward

def solve(edges,total):
    flow=cost=0
    while flow<total:
        distance={'START':0};parent={};pending=deque(['START']);queued={'START'}
        while pending:
            a=pending.popleft();queued.remove(a)
            for i,(b,capacity,value,reverse) in enumerate(edges[a]):
                candidate=distance[a]+value
                if capacity and candidate<distance.get(b,float('inf')):
                    distance[b]=candidate;parent[b]=(a,i)
                    if b not in queued:queued.add(b);pending.append(b)
        if 'END' not in distance:break
        amount=total-flow;b='END'
        while b!='START':a,i=parent[b];amount=min(amount,edges[a][i][1]);b=a
        b='END'
        while b!='START':
            a,i=parent[b];forward=edges[a][i];forward[1]-=amount;edges[b][forward[3]][1]+=amount;b=a
        flow+=amount;cost+=amount*distance['END']
    return flow,cost

def allocate_quality(counts,total):
    # 8 * turns - 10 * short answers <= 0 is exactly >=80% short answers.
    edges=defaultdict(list);tracked={}
    for source in sorted({s for s,c,cost in counts}):add_edge(edges,'START','S:'+source,source_cap(source,total),0)
    for key,amount in sorted(counts.items()):
        source,category,cost=key;tracked[key]=add_edge(edges,'S:'+source,'C:'+category,amount,cost)
    for category,n in CATEGORIES.items():add_edge(edges,'C:'+category,'END',n*total//100000,0)
    flow,cost=solve(edges,total)
    return flow,{key:counts[key]-entry[1] for key,entry in tracked.items()},cost

def stratified_subset(counts):
    """Round each source/category/length stratum to one fifth, with hard caps."""
    base={key:n//5 for key,n in counts.items()};sources=defaultdict(int);taken=defaultdict(int);cats=defaultdict(int)
    for (source,category,short,cost),n in counts.items():sources[source]+=n
    for (source,category,short,cost),n in base.items():taken[source]+=n;cats[category]+=n
    edges=defaultdict(list);tracked={};total=20000-sum(base.values())
    for source,n in sorted(sources.items()):
        add_edge(edges,'START','S:'+source,min(source_cap(source,20000),(n+4)//5)-taken[source],0)
    # Answer quality takes precedence over fractional remainders. The scale
    # makes even a one-unit quality improvement dominate every remainder tie.
    scale=4*total+1
    for key,n in sorted(counts.items()):
        if n%5:tracked[key]=add_edge(edges,'S:'+key[0],'C:'+key[1],1,key[3]*scale-(n%5))
    for category,n in CATEGORIES.items():add_edge(edges,'C:'+category,'END',n//5-cats[category],0)
    flow,_=solve(edges,total)
    if flow!=total:raise ValueError('Cannot round source/category/length strata under source caps')
    for key,entry in tracked.items():base[key]+=1-entry[1]
    if sum(key[3]*n for key,n in base.items())>0:raise ValueError('Comparison subset cannot meet 80% short answers')
    return base
