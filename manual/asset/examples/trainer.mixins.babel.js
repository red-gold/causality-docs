import { causalNetSGDOptimizer, TrainerMixins } from 'causal-net.optimizers';
import { platform } from 'causal-net.utils';
import { Tensor } from 'causal-net.core';
const PipeLineConfigure = {
    Trainer: {
        Optimizer: causalNetSGDOptimizer.adam,
        OptimizerParameters: {},
        TrainableParameters: [],
    }
}
class SimplePipeline extends platform.mixWith(Tensor, [TrainerMixins]){
    constructor(configure){
        super();
        this.Trainer = configure.Trainer;
    }
}
let pipeline = new SimplePipeline(PipeLineConfigure);
console.log([ pipeline.Optimizer instanceof Tensor ]);
console.log([ pipeline.OptimizerParameters, pipeline.TrainableParameters ]);