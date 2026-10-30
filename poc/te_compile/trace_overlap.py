"""Recompute comm/compute overlap from saved rank-0 chrome traces (GPU kernels only).

Usage: python trace_overlap.py <trace.json>... ; 3 profiled steps per trace (overlap_stats).
"""
import collections
import json
import sys

def union(iv):
    m=[]
    for s,e in sorted(iv):
        if m and s<=m[-1][1]: m[-1][1]=max(m[-1][1],e)
        else: m.append([s,e])
    return m
def L(iv): return sum(e-s for s,e in iv)
def inter(a,b):
    o=[];i=j=0
    while i<len(a) and j<len(b):
        s,e=max(a[i][0],b[j][0]),min(a[i][1],b[j][1])
        if s<e:o.append([s,e])
        if a[i][1]<b[j][1]: i+=1
        else: j+=1
    return o
ITERS=3
for f in sorted(sys.argv[1:]):
    ev=json.load(open(f))
    ev=ev['traceEvents'] if isinstance(ev,dict) else ev
    k=[e for e in ev if e.get('cat')=='kernel' and e.get('dur',0)>0]
    comm=[(e['ts'],e['ts']+e['dur']) for e in k if 'nccl' in e['name'].lower()]
    comp=[(e['ts'],e['ts']+e['dur']) for e in k if 'nccl' not in e['name'].lower()]
    cu,pu=union(comm),union(comp); bu=union(comm+comp)
    ov=L(inter(cu,pu))
    streams=collections.Counter((e['tid'],'nccl' in e['name'].lower()) for e in k)
    ms=lambda x: round(x/1000/ITERS,2)
    name=f.split('results/')[1].replace('/graphs/trace-',' ').replace('.json','')
    print(f"{name:70s} comm={ms(L(cu))} overlapped={ms(ov)} exposed={ms(L(cu)-ov)} compute={ms(L(pu))} busy={ms(L(bu))} span={ms(bu[-1][1]-bu[0][0])} nccl_streams={sorted({t for (t,c) in streams if c})}")
