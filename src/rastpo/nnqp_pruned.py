"""KKT-checked coordinate pruning with a full NNQP solver as fallback.

Deleting negative coordinates is only a proposal. The candidate is used only
after primal, dual and complementarity checks, with a conservative boundary
screen.
"""
import torch


def solve(S,mu,reference,**kwargs):
    B,k,_=S.shape
    if B>kwargs.get('cap',8192):
        return reference(S,mu,**kwargs)
    with torch.no_grad():
        v=torch.linalg.solve(S,mu[...,None]).squeeze(-1)
        interior=(v>=0).all(-1)
        if bool(interior.all()):return v
        rows=(~interior).nonzero().squeeze(-1)
        A=S[rows];b=mu[rows];x=v[rows]
        keep=torch.ones_like(b,dtype=torch.bool)
        for _ in range(k):
            bad=keep&(x<=0)
            if not bool(bad.any()):break
            keep=keep&~bad
            mask=keep.to(A.dtype)
            system=A*mask[...,None]*mask[:,None,:]+torch.diag_embed(1-mask)
            x=torch.linalg.solve(system,(b*mask)[...,None]).squeeze(-1)*mask
        fitted=A.bmm(x[...,None]).squeeze(-1)
        dual=fitted-b
        eps=torch.finfo(A.dtype).eps;tiny=torch.finfo(A.dtype).tiny
        scale=torch.maximum(b.abs().amax(-1),fitted.abs().amax(-1)).clamp_min(tiny)
        tol=128*eps*scale
        primal=(x>=0).all(-1)
        stationarity=torch.where(keep,dual.abs(),torch.zeros_like(dual)).amax(-1)
        dual_min=torch.where(~keep,dual,torch.full_like(dual,float('inf'))).amin(-1)
        # Boundary cases use the full NNQP solver.
        rel=eps**0.5
        primal_scale=x.abs().amax(-1,keepdim=True).clamp_min(tiny)
        boundary=(keep&(x<=rel*primal_scale))|((~keep)&(dual<=rel*scale[:,None]))
        zero=(b==0).all(-1)
        passed=primal&(stationarity<=tol)&(dual_min>=-tol)&(~boundary.any(-1)|zero)
        answer=v.clone();answer[rows]=x
        if bool((~passed).any()):
            answer[rows[~passed]]=reference(A[~passed],b[~passed],**kwargs)
        return answer
