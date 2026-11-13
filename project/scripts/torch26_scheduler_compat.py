"""Bridge deprecated Torch2.6 verbose=False scheduler calls to Torch2.7.

No learning-rate parameters or optimizer state are modified. verbose=True fails
explicitly rather than silently dropping a user-requested logging behavior.
"""
import inspect

def install():
    import torch.optim.lr_scheduler as module
    patched=[]
    def adapt(original):
        class Compatible(original):
            def __init__(self,*args,verbose=False,**kwargs):
                if verbose:raise ValueError('Compatibility bridge supports only verbose=False')
                super().__init__(*args,**kwargs)
        Compatible.__name__=original.__name__
        Compatible.__qualname__=original.__qualname__
        return Compatible
    for name in ['LinearLR','ConstantLR','StepLR','MultiStepLR','ExponentialLR',
                 'CosineAnnealingLR','CosineAnnealingWarmRestarts','PolynomialLR',
                 'ReduceLROnPlateau','SequentialLR','ChainedScheduler']:
        original=getattr(module,name,None)
        if original is not None and 'verbose' not in inspect.signature(original).parameters:
            setattr(module,name,adapt(original));patched.append(name)
    return patched

def verify():
    import torch
    import torch.optim.lr_scheduler as module
    original=module.LinearLR
    p=torch.nn.Parameter(torch.tensor(1.))
    opt=torch.optim.SGD([p],lr=.1)
    sched=original(opt,start_factor=.1,end_factor=1,total_iters=5)
    expected=[opt.param_groups[0]['lr']]
    for _ in range(6):opt.step();sched.step();expected.append(opt.param_groups[0]['lr'])
    patched=install()
    q=torch.nn.Parameter(torch.tensor(1.))
    opt2=torch.optim.SGD([q],lr=.1)
    sched2=module.LinearLR(opt2,start_factor=.1,end_factor=1,total_iters=5,verbose=False)
    actual=[opt2.param_groups[0]['lr']]
    for _ in range(6):opt2.step();sched2.step();actual.append(opt2.param_groups[0]['lr'])
    if expected!=actual:raise RuntimeError('Scheduler numeric behavior changed')
    return {'patched':patched,'expected':expected,'actual':actual,'passed':True}

if __name__=='__main__':
    import json
    print(json.dumps(verify()))
