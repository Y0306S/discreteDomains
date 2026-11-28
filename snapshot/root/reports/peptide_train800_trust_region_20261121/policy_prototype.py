"""Isolated sequence-first search building blocks. Not imported by production.

No oracle is invoked here. Callers must count every decoded candidate, rejection,
cache hit and eventual MIC request separately. Acquisition scores are supplied
by a callback operating on canonical encodings of the actual decoded peptides.
"""
from dataclasses import dataclass
from Levenshtein import distance
import numpy as np
AA=frozenset('ACDEFGHIKLMNPQRSTVWY')
def syntax(s):return bool(s) and len(s)<=30 and s.count('C')<=1 and not(set(s)-AA)
def centers(reference,observed_sequences,observed_scores,maximum=4,promising_count=32):
 if not syntax(reference) or maximum<1 or promising_count<1:raise ValueError('Invalid reference or center count')
 if len(observed_sequences)!=len(observed_scores):raise ValueError('Observation lengths differ')
 feasible={}
 for s,y in zip(observed_sequences,observed_scores):
  if not np.isfinite(y):raise ValueError('Nonfinite observed score')
  if syntax(s) and distance(s,reference)<=len(reference)//4:
   feasible[s]=max(feasible.get(s,-np.inf),float(y))
 if not feasible:raise ValueError('A measured feasible incumbent is required')
 # Objective scores are maximized (-MIC). Keep incumbent and reference, then
 # greedily spread centers in sequence space among already measured candidates.
 selected=[max(feasible,key=lambda s:(feasible[s],s))]
 if len(selected)<maximum and reference not in selected:selected.append(reference)
 while len(selected)<maximum:
  promising=sorted(feasible,key=lambda s:(feasible[s],s),reverse=True)[:promising_count]
  candidates=[s for s in promising if s not in selected]
  if not candidates:break
  selected.append(max(candidates,key=lambda s:(min(distance(s,c) for c in selected),feasible[s],s)))
 return selected
@dataclass(frozen=True)
class Candidate:
 sequence:str
 center_index:int
@dataclass(frozen=True)
class RankedCandidate:
 decoded_index:int
 sequence:str
 acquisition_score:float
 canonical_feature:tuple

def select(reference,search_centers,decoded,seen,local_edits,encode,acquisition_scores,batch_size):
 """Filter actual sequences, then reencode, rank, preserving decoded indices.

 local_edits=None is the explicit exploration arm, still globally constrained.
 A short accepted pool produces a short batch; no hidden oracle calls or padding.
 """
 if local_edits not in (None,2,3) or batch_size<1:raise ValueError('Invalid policy')
 if not syntax(reference) or not search_centers:raise ValueError('Invalid reference/centers')
 if any(not syntax(c) or distance(c,reference)>len(reference)//4 for c in search_centers):raise ValueError('Infeasible center')
 known=set(seen);accepted=[];rejected=[]
 for i,item in enumerate(decoded):
  if not isinstance(item.center_index,int) or not 0<=item.center_index<len(search_centers):raise ValueError('Unknown origin center')
  s=item.sequence;why=None
  if not syntax(s):why='sequence_rule'
  elif distance(s,reference)>len(reference)//4:why='reference_limit'
  elif local_edits is not None and distance(s,search_centers[item.center_index])>local_edits:why='local_limit'
  elif s in known:why='cached_or_duplicate'
  if why:rejected.append((i,why))
  else:accepted.append((i,s));known.add(s)
 if not accepted:return [],rejected
 features=np.asarray(encode([s for _,s in accepted]),dtype=float)
 if features.ndim!=2 or len(features)!=len(accepted) or not np.isfinite(features).all():raise ValueError('Invalid canonical encodings')
 scores=np.asarray(acquisition_scores(features),dtype=float)
 if scores.shape!=(len(accepted),) or not np.isfinite(scores).all():raise ValueError('Invalid acquisition scores')
 order=sorted(range(len(accepted)),key=lambda i:(-scores[i],accepted[i][0]))[:batch_size]
 return [RankedCandidate(accepted[i][0],accepted[i][1],float(scores[i]),tuple(features[i])) for i in order],rejected
