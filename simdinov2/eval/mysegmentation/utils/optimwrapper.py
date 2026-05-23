
import torch._dynamo as dynamo
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS

@OPTIM_WRAPPERS.register_module()
class OptimWrapperNoDynamo(OptimWrapper):
    @dynamo.disable
    def _step_no_compile(self):
        return self.optimizer.step()

    def step(self):
        return self._step_no_compile()
