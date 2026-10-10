import numpy as np
from hs import *
for name, lam, e2, pol in [("Ag",633.,-18.3+0.48j,1),("Au",800.,-24.1+1.5j,1),("glass",800.,2.25,2)]:
  for Q in (10.,3.):
    k0 = 2*PI/(lam*(1+1j/(2*Q)))
    X,Z = -300., 210.
    ref = g_ind(X,Z,k0,1.0,e2,pol,*make_path(k0,1.0,e2,pol,Z,300.,n_hump=200)[:2])[0]
    row=[]
    for d in [0.2,0.4,0.8,1.2]:
      for nh in [24,48,96]:
        q,wq,info = make_path(k0,1.0,e2,pol,Z,300.,n_hump=nh,delta_frac=d)
        row.append(f"d{d}/n{nh}:{abs(g_ind(X,Z,k0,1.0,e2,pol,q,wq)[0]-ref)/abs(ref):.0e}")
    print(name,pol,Q," ".join(row))
