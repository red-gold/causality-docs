import { causalNetSGDOptimizer, TrainerMixins, EvaluatorMixins } from 'causal-net.optimizers';
import { causalNetModels, ModelMixins } from 'causal-net.models';
import { causalNetParameters, causalNetLayers, causalNetRunner, LayerRunnerMixins } from 'causal-net.layer';
import { causalNetCore, Functor } from 'causal-net.core';
import { platform } from 'causal-net.utils';
import { Tensor } from 'causal-net.core';
import { termLogger, LoggerMixins } from 'causal-net.log';

class SimplePipeline extends platform.mixWith(Tensor, [ 
        LayerRunnerMixins, 
        ModelMixins, 
        EvaluatorMixins,
        LoggerMixins,
        TrainerMixins]){
    constructor( netRunner, functor, logger){
        super();
        this.F = functor;
        this.LayerRunner = netRunner;
        this.Logger = logger;
    }
}
const T = causalNetCore.CoreTensor;
const R = causalNetCore.CoreFunctor;
const F = new Functor();
const DummyData = (batchSize)=>{
    let samples = [  [[0], [1], [2], [3]], 
                     [[0], [1], [2], [3]], 
                     [[0], [1], [2], [3]] ];
    let labels  = [ [0,1] ];
    return [{samples, labels}];
}
console.log(DummyData(1));
(async ()=>{
    let convLayer = causalNetLayers.convolution({ kernelSize: [2, 2], 
                                                  filters: [1, 2], 
                                                  flatten: true } );
    let denseLayer = causalNetLayers.dense({ inputSize: 8, outputSize: 2 });
    const PipeLineConfigure = {
        Dataset: {
            TrainDataGenerator: DummyData,
            TestDataGenerator: DummyData,
        },
        Net: { 
                Parameters: causalNetParameters.InitParameters(),
                Layers: { Predict: [ convLayer, denseLayer ] },
                Model: causalNetModels.classification(2),
                Optimizer: causalNetSGDOptimizer.adam({learningRate: 0.01})
        }
    };

    let pipeline = new SimplePipeline( causalNetRunner, F, termLogger);
    pipeline.setByConfig(PipeLineConfigure);
    const NumEpochs = 10, BatchSize = 1;
    console.log(await pipeline.train(NumEpochs, BatchSize));
    console.log(await pipeline.test());
})();


