# EANN-KDD18


[EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/citation.cfm?id=3219819.3219903)  
 [Yaqing Wang](http://www.acsu.buffalo.edu/~yaqingwa/),
 [Fenglong Ma](http://www.acsu.buffalo.edu/~fenglong/), 
 Zhiwei Jin, Ye Yuan, 
 Guangxu Xun,
 [Kishlay Jha](http://people.virginia.edu/~kj6ww/),
  [Lu Su](https://cse.buffalo.edu/~lusu/),
 [Jing Gao](https://cse.buffalo.edu/~jing/)
 
 SUNY Buffalo. KDD, 2018.
 
## Files
 The __data__ folder contains the partial dataset. The train_id, validate_id and test_id are event id. 
 
 
 The __src__ folder contains the data preprocessing file: __process_data_weibo.py__, and the model files: __EANN.py__ and __EANN_text.py__.
 EANN.py is for text and image multimodal features and EANN_text.py is only using textual featues.
 

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
