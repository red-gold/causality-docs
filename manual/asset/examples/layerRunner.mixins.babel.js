import { causalNetParameters, causalNetLayers, causalNetRunner, LayerRunnerMixins  } from 'causal-net.layer';
import { causalNetCore } from 'causal-net.core';
import { platform } from 'causal-net.utils';
import { Tensor } from 'causal-net.core';
import { termLogger } from 'causal-net.log';
const PipeLineConfigure = {
    Net: { 
            Parameters: causalNetParameters.InitParameters(),
            Layers: { 
                Predict: [  causalNetLayers.dense({inputSize:4,outputSize:2}) ], 
                Encode: [ causalNetLayers.dense({inputSize:4,outputSize:2}) ], 
                Decode: [ causalNetLayers.dense({inputSize:4,outputSize:2}) ] 
            }
    }
}
class SimplePipeline extends platform.mixWith(Tensor, [ LayerRunnerMixins ]){
    constructor(layerRunner, logger){
        super();
        this.logger = logger;
        this.LayerRunner = layerRunner;
    }
}
const T = causalNetCore.CoreTensor;
(async ()=>{
    let pipeline = new SimplePipeline(causalNetRunner, termLogger);
    pipeline.setByConfig(PipeLineConfigure);
    const { Predictor, Encoder, Decoder } = pipeline.LayerRunner;
    console.log({ Predictor, Encoder, Decoder });   
})().catch(err=>{
    console.error({err});
});