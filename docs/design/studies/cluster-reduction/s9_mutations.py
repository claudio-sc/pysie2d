"""Break the code on purpose: a gate that still passes is not a gate."""
import pytest, sys, warnings, numpy as np
import pysie2d.cluster as C, pysie2d.kernels as K
TF="tests/test_cluster_eigen_eps.py"
def run(name, setup, teardown):
    setup()
    class Rec:
        failed=[]; 
        def pytest_runtest_logreport(self,report):
            if report.when=="call" and report.failed: self.failed.append(report.nodeid.split("::")[1].split("[")[0])
    r=Rec(); r.failed=[]
    pytest.main(["-q","-W","ignore","--no-header","--tb=no","-p","no:cacheprovider",TF,"--co" if False else "-q"],plugins=[r])
    teardown()
    print(f"\n@@ {name}: {len(r.failed)} failed -> {sorted(set(r.failed))}\n",flush=True)
ub=C._upper_branch; d=C.ClusterEigenEps._reduced_derivative; mv=C.ClusterEigenEps._mode_vector; fac=C.ClusterEigenEps._factor; ext=C.ClusterBIESolver._exterior_rows
rs=C.ClusterEigenEps._residual
def st(n,v): return lambda: setattr(C,n,v)
run("rank tol 1e-3",st("COUPLING_RANK_TOL",1e-3),st("COUPLING_RANK_TOL",1e-13))
run("rank tol 1e-7",st("COUPLING_RANK_TOL",1e-7),st("COUPLING_RANK_TOL",1e-13))
run("principal branch",st("_upper_branch",lambda e:np.sqrt(complex(e))),st("_upper_branch",ub))
run("derivative sign flipped",lambda:setattr(C.ClusterEigenEps,"_reduced_derivative",lambda s,e:-d(s,e)),lambda:setattr(C.ClusterEigenEps,"_reduced_derivative",d))
run("mode vector wrong sign on particle 1",lambda:setattr(C.ClusterEigenEps,"_mode_vector",lambda s,e,y:(lambda x:(x.__setitem__(s.solver.cluster.slice(len(s.solver.cluster)-1),-x[s.solver.cluster.slice(len(s.solver.cluster)-1)]),x)[1])(mv(s,e,y))),lambda:setattr(C.ClusterEigenEps,"_mode_vector",mv))
def noreflect(self,q,w,refl): return ext(self,q,w,None)
run("substrate dropped from exterior rows",lambda:setattr(C.ClusterBIESolver,"_exterior_rows",noreflect),lambda:setattr(C.ClusterBIESolver,"_exterior_rows",ext))
run("no refinement",st("MAX_REFINE_STEPS",0),st("MAX_REFINE_STEPS",3))
