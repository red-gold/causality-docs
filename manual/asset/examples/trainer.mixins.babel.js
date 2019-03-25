import { causalNetSGDOptimizer, TrainerMixins } from 'causal-net.optimizers';
import { causalNetModels, ModelMixins } from 'causal-net.models';
import { causalNetParameters, causalNetLayers, causalNetRunner, NetMixins } from 'causal-net.layer';
import { causalNetCore } from 'causal-net.core';
import { platform } from 'causal-net.utils';
import { Tensor } from 'causal-net.core';
import { termLogger } from 'causal-net.log';

class SimplePipeline extends platform.mixWith(Tensor, [ 
        NetMixins, 
        ModelMixins, 
        TrainerMixins]){
    constructor(netParameters, netRunner, logger){
        super();
        this.logger = logger;
        this.Parameters = netParameters;
        this.Net = netRunner;
    }
}
const T = causalNetCore.CoreTensor;
(async ()=>{
    const PipeLineConfigure = {
        Dataset: {
            TrainData: [[1,2,3,4]],
            TestData: [[1,2,3,4]],
            ValidateData: [[1,2,3,4]]
        },
        Net: { 
                Parameters: { Predict: null, Encode: null, Decode: null },
                Layers: { 
                    Predict: [ causalNetLayers.dense(4, 3), 
                               causalNetLayers.dense(3, 2) ], 
                    Encode: [ causalNetLayers.dense(4, 2) ], 
                    Decode: [ causalNetLayers.dense(4, 2) ] 
                },
                Model: causalNetModels.classification(2),
                Optimizer: causalNetSGDOptimizer.adam({learningRate: 0.01})
        }
    };

    let pipeline = new SimplePipeline( causalNetParameters, 
                            causalNetRunner, termLogger);
    pipeline.setByConfig(PipeLineConfigure);
    const { Predictor } = pipeline.Net;
    let predictInfer = Predictor(T.tensor([[1,2,3,4]]));
    predictInfer.print();
    predictInfer = pipeline.PredictModel(T.tensor([[1,2,3,4]]));
    predictInfer.print();
    
    let modelOneHotPredict = pipeline.OneHotPredictModel(T.tensor([[1,2,3,4]]).asType('float32'));
    modelOneHotPredict.print();
    let fit = pipeline.FitModel(T.tensor([[1,2,3,4]]).asType('float32'));
    fit.print();
    let modelLoss = pipeline.LossModel(T.tensor([[1,2,3,4]]).asType('float32'), 
                             T.tensor([[0, 1]]).asType('float32'));
    modelLoss.print();
    let trainLoss = pipeline.Trainer(T.tensor([[1,2,3,4]]).asType('float32'), 
        T.tensor([[0, 1]]).asType('float32'));
    trainLoss.print();
    trainLoss = pipeline.Trainer(T.tensor([[1,2,3,4]]).asType('float32'), 
        T.tensor([[0, 1]]).asType('float32'));
    trainLoss.print();
})().catch(err=>{
    console.error({err});
});