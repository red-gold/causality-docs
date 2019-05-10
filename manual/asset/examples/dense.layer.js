import { causalNetLayers } from 'causal-net.layer';
let denseLayer = causalNetLayers.dense({inputSize:3,outputSize:2});
console.log({denseLayer: denseLayer.Config});
