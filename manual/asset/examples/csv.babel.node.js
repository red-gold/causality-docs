import { csvUtils } from 'causal-net.utils';
(async ()=>{
    let data = await csvUtils.readCSV('../../datasets/credict.csv');
    console.log(data);
})();
