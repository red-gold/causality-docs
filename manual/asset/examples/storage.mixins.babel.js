import { StorageMixins, indexDBStorage } from 'causal-net.storage';
import { platform } from 'causal-net.utils';
import { Tensor, Store } from 'causal-net.core';

class SimplePipeline extends platform.mixWith(Tensor, [StorageMixins]){
    constructor(){
        super();
        this.Storage = indexDBStorage;
    }
}
let pipeline = new SimplePipeline();
console.log(pipeline.Storage instanceof Store);