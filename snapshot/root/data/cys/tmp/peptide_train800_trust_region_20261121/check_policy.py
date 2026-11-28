"""Boundary and coordinate-alignment checks for proposed isolated sequence gate."""
import numpy as np
from Levenshtein import distance
from policy_prototype import Candidate,centers,select
ref='AKLAFKLAKKLA'
a='RKLAFKLAKKLA';b='AKLRFKLAKKLA';far='RRLRFKLAKKLA'
cs=centers(ref,[ref,a,b,far,'WWWWWWWWWWWW'],[-100,-70,-90,-50,-1])
assert cs[0]==far and ref in cs and 'WWWWWWWWWWWW' not in cs
# The encoder sees only surviving actual peptides, never raw proposed latents.
called=[]
def encode(s):called.append(s);return np.array([[11 if x==a else 17,2] for x in s])
def rank(x):assert np.array_equal(x[:,1],np.array([2,2]));return -x[:,0]
candidates=[Candidate('WWWWWWWWWWWW',0),Candidate(a,0),Candidate(a,0),Candidate(b,0),Candidate(ref,0)]
selected,rejected=select(ref,[ref],candidates,{ref},2,encode,rank,50)
assert called==[[a,b]] and [r.decoded_index for r in selected]==[1,3]
assert [r.sequence for r in selected]==[a,b] and [r[0] for r in rejected]==[0,2,4]
assert selected[0].canonical_feature==(11,2)
# Local3 adds globally valid candidates that local2 rejects.
assert distance(far,ref)==3
for k,expected in [(2,0),(3,1),(None,1)]:
 selected,_=select(ref,[ref],[Candidate(far,0)],set(),k,lambda s:np.ones((len(s),256)),lambda x:np.ones(len(x)),50)
 assert len(selected)==expected
# Indels and strict global boundary remain enforced, even in exploration mode.
for s,ok in [(ref+'A',True),(ref[:-1],True),('RRLRFKLAKKLW',False),('CC'+ref,False)]:
 selected,_=select(ref,[ref],[Candidate(s,0)],set(),None,lambda s:np.ones((len(s),256)),lambda x:np.ones(len(x)),1)
 assert bool(selected)==ok
# Empty pool does not invoke encoder/acquisition or silently fill a batch.
def must_not_run(*args):raise AssertionError('Called on empty pool')
assert select(ref,[ref],[Candidate(ref,0)],{ref},2,must_not_run,must_not_run,50)[0]==[]
for fn in [lambda:select(ref,[ref],[Candidate(a,8)],set(),2,encode,rank,1),lambda:select(ref,[ref],[Candidate(a,0)],set(),2,lambda s:np.ones((2,2)),rank,1)]:
 try:fn()
 except ValueError:pass
 else:raise AssertionError('Invalid alignment was accepted')
print('PASS: feasible multi-centers,2vs3,global boundary,indels,dedup,canonical encoding,ranking index alignment,short-batch accounting,malformed inputs')
