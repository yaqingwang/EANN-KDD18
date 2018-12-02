# EANN-KDD18


[EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/citation.cfm?id=3219819.3219903)  
 [Yaqing Wang](http://www.acsu.buffalo.edu/~yaqingwa/),
 [Fenglong Ma](http://www.acsu.buffalo.edu/~fenglong/), 
 Zhiwei Jin, Ye Yuan, 
 Guangxu Xun,
 Kishlay Jha,
  [Lu Su](https://cse.buffalo.edu/~lusu/),
 [Jing Gao](https://cse.buffalo.edu/~jing/)
 
 SUNY Buffalo. KDD, 2018.
 
 
 
 ## Main Idea
One of the unique challenges for fake news detection on social media is how to identify fake news on  **newly emerged events**. The EANN is desgined to  __extract shared features among all events__ to effectively improve the performance of fake news detection on never-seen events.


## Experiment
Comparision between reduced model (w/o adversarial) and EANN(w adversarial)

<img src="https://github.com/yaqingwang/EANN-KDD18/blob/master/Fig/Accuracy.png" width="300">  <img src="https://github.com/yaqingwang/EANN-KDD18/blob/master/Fig/F1.png" width="300">

The feature representations learned by the proposed model EANN (right) are more discriminable than fake news detection (w/o adv).

<img src="https://github.com/yaqingwang/EANN-KDD18/blob/master/Fig/baseline_tsne.png" width="256">  <img src="https://github.com/yaqingwang/EANN-KDD18/blob/master/Fig/model_tsne.png" width="256">
 
 

 ## Citation
If you use this code for your research, please cite our [paper](https://dl.acm.org/citation.cfm?id=3219819.3219903):

```
@inproceedings{wang2018eann,
  title={EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection},
  author={Wang, Yaqing and Ma, Fenglong and Jin, Zhiwei and Yuan, Ye and Xun, Guangxu and Jha, Kishlay and Su, Lu and Gao, Jing},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={849--857},
  year={2018},
  organization={ACM}
}
```
