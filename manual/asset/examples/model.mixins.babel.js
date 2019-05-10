import { causalNetModels, ModelMixins } from 'causal-net.models';
import { causalNetParameters, causalNetLayers, causalNetRunner, LayerRunnerMixins } from 'causal-net.layer';
import { causalNetCore } from 'causal-net.core';
import { platform } from 'causal-net.utils';
import { Tensor } from 'causal-net.core';
import { termLogger, LoggerMixins } from 'causal-net.log';

class SimplePipeline extends platform.mixWith(Tensor, [LayerRunnerMixins, ModelMixins, LoggerMixins]){
    constructor(netRunner, logger){
        super();
        this.Logger = logger;
        this.LayerRunner = netRunner;
    }
}
                  
    
const T = causalNetCore.CoreTensor;
(async ()=>{
    let convLayer = causalNetLayers.convolution({kernelSize:[2,2], filters:[1,2], flatten:true} );
    let denseLayer = causalNetLayers.dense({inputSize:24,outputSize:2});
    const PipeLineConfigure = {
        Net: { 
                Parameters: causalNetParameters.InitParameters(),
                Layers: { 
                    Predict: [  convLayer, denseLayer], 
                    Encode: [ causalNetLayers.dense({inputSize:24,outputSize:2}) ], 
                    Decode: [ causalNetLayers.dense({inputSize:24,outputSize:2}) ] 
                },
                Model: causalNetModels.classification(2)
        }
    };

    let pipeline = new SimplePipeline( causalNetRunner, termLogger);
    pipeline.setByConfig(PipeLineConfigure);
    
    let inputTensor = T.tensor([  [1,2,3,4],
                                  [1,2,3,4],
                                  [1,2,3,4] ]).reshape([1,3,4,1]);
    let modelOneHotPredict = pipeline.OneHotPredictModel(inputTensor);
    modelOneHotPredict.print();
    let fit = pipeline.FitModel(inputTensor);
    fit.print();
    let modelLoss = pipeline.LossModel(inputTensor, 
                             T.tensor([[0, 1]]).asType('float32'));
    modelLoss.print();
})().catch(err=>{
    console.error({err});
});