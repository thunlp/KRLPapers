## Must-read papers on KRL/KE.
KRL: knowledge representation learning. KE: knowledge embedding.

Contributed by [Shulin Cao](https://github.com/ShulinCao) and [Xu Han](https://github.com/THUCSTHanxu13).

We release [OpenKE](https://github.com/thunlp/openKE), an open source toolkit for KRL/KE. This repository provides a standard KRL/KE training and testing framework. Currently, the implemented models in OpenKE include TransE, TransH, TransR, TransD, RESCAL, DistMult, ComplEx and HolE.

### Survey papers:

1. **Representation Learning: A Review and New Perspectives.**
*Yoshua Bengio, Aaron Courville, and Pascal Vincent.* IEEE 2013. [paper](https://arxiv.org/pdf/1206.5538)

1. **Knowledge Representation Learning: A Review. (In Chinese)**
*Zhiyuan Liu, Maosong Sun, Yankai Lin, Ruobing Xie.* 2016. [paper](http://crad.ict.ac.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=3099)

1. **A Review of Relational Machine Learning for Knowledge Graphs.**
*Maximilian Nickel, Kevin Murphy, Volker Tresp, Evgeniy Gabrilovich.* IEEE 2016. [paper](https://arxiv.org/pdf/1503.00759.pdf)

1. **Knowledge Graph Embedding: A Survey of Approaches and Applications.**
*Quan Wang, Zhendong Mao, Bin Wang, Li Guo.* IEEE 2017.  [paper](http://ieeexplore.ieee.org/abstract/document/8047276/)

### Journal and Conference papers:

1. **RESCAL: A Three-Way Model for Collective Learning on Multi-Relational Data.**
*Nickel Maximilian, Tresp Volker, Kriegel Hans-Peter.* ICML 2011. [paper](http://www.icml-2011.org/papers/438_icmlpaper.pdf) [code](https://github.com/thunlp/OpenKE)

1. **SE: Learning Structured Embeddings of Knowledge Bases.**
*Antoine Bordes, Jason Weston, Ronan Collobert, Yoshua Bengio.* AAAI 2011. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898)

1. **LFM: A Latent Factor Model for Highly Multi-relational Data.**
*Rodolphe Jenatton, Nicolas L. Roux, Antoine Bordes, Guillaume R. Obozinski.* NIPS 2012. [paper](http://papers.nips.cc/paper/4744-a-latent-factor-model-for-highly-multi-relational-data.pdf)

1. **NTN: Reasoning With Neural Tensor Networks for Knowledge Base Completion.**
*Richard Socher, Danqi Chen, Christopher D. Manning, Andrew Ng.* NIPS 2013. [paper](http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf)

1. **TransE: Translating Embeddings for Modeling Multi-relational Data.**
*Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko.*  NIPS 2013. [paper](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) [code](https://github.com/thunlp/OpenKE)

1. **TransH: Knowledge Graph Embedding by Translating on Hyperplanes.**
*Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen.* AAAI 2014. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546) [code](https://github.com/thunlp/OpenkE)

1. **TransR & CTransR: Learning Entity and Relation Embeddings for Knowledge Graph Completion.**
*Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu.* AAAI 2015. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/) [KB2E](https://github.com/thunlp/KB2E) [OpenKE](https://github.com/thunlp/OpenKE)

1. **TransD: Knowledge Graph Embedding via Dynamic Mapping Matrix.**
*Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao.* ACL 2015. [paper](http://anthology.aclweb.org/P/P15/P15-1067.pdf) [KB2E](https://github.com/thunlp/KB2E) [OpenKE](https://github.com/thunlp/OpenKE)

1. **TransA: An Adaptive Approach for Knowledge Graph Embedding.**
*Han Xiao, Minlie Huang, Hao Yu, Xiaoyan Zhu.* arXiv 2015. [paper](https://arxiv.org/pdf/1509.05490.pdf)

1. **KG2E: Learning to Represent Knowledge Graphs with Gaussian Embedding.**
*Shizhu He, Kang Liu, Guoliang Ji and Jun Zhao.* CIKM 2015. [paper](https://pdfs.semanticscholar.org/941a/d7796cb67637f88db61e3d37a47ab3a45707.pdf) [code](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/code/cikm15_he_code.zip)

1. **DistMult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases.**
*Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng.* ICLR 2015. [paper](https://arxiv.org/pdf/1412.6575) [code](https://github.com/thunlp/OpenKE)

1. **PTransE: Modeling Relation Paths for Representation Learning of Knowledge Bases.**
*Yankai Lin, Zhiyuan Liu, Huanbo Luan, Maosong Sun, Siwei Rao, Song Liu.* EMNLP 2015. [paper](https://arxiv.org/pdf/1506.00379.pdf) [code](https://github.com/thunlp/KB2E)

1. **RTransE: Composing Relationships with Translations.**
*Alberto García-Durán, Antoine Bordes, Nicolas Usunier.* EMNLP 2015. [paper](http://www.aclweb.org/anthology/D15-1034.pdf)

1. **ManifoldE: From One Point to A Manifold: Knowledge Graph Embedding For Precise Link Prediction.**
*Han Xiao, Minlie Huang and Xiaoyan Zhu.* IJCAI 2016. [paper](https://arxiv.org/pdf/1512.04792.pdf)

1. **TransG: A Generative Mixture Model for Knowledge Graph Embedding.**
*Han Xiao, Minlie Huang, Xiaoyan Zhu.* ACL 2016. [paper](http://www.aclweb.org/anthology/P16-1219) [code](https://github.com/BookmanHan/Embedding)

1. **ComplEx: Complex Embeddings for Simple Link Prediction.**
*Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and Guillaume Bouchard.* ICML 2016. [paper](http://proceedings.mlr.press/v48/trouillon16.pdf) [code](https://github.com/ttrouill/complex) [OpenKE](https://github.com/thunlp/OpenKE)

1. **HolE: Holographic Embeddings of Knowledge Graphs.**
*Maximilian Nickel, Lorenzo Rosasco, Tomaso A. Poggio.* AAAI 2016. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12484/11828) [code](https://github.com/mnick/holographic-embeddings) [OpenKE](https://github.com/thunlp/OpenKE)

1. **KR-EAR: Knowledge Representation Learning with Entities, Attributes and Relations.**
*Yankai Lin, Zhiyuan Liu, Maosong Sun.* IJCAI 2016. [paper](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/ijcai2016_krear.pdf) [code](https://github.com/thunlp/KR-EAR) 

1. **TranSparse: Knowledge Graph Completion with Adaptive Sparse Transfer Matrix.**
*Guoliang Ji, Kang Liu, Shizhu He, Jun Zhao.* AAAI 2016. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11982/11693) [code](https://github.com/thunlp/Fast-TransX)

1. **TKRL: Representation Learning of Knowledge Graphs with Hierarchical Types.**
*Ruobing Xie, Zhiyuan Liu, Maosong Sun.* IJCAI 2016. [paper](http://www.thunlp.org/~lzy/publications/ijcai2016_tkrl.pdf) [code](https://github.com/thunlp/TKRL)

1. **STransE: A Novel Embedding Model of Entities and Relationships in Knowledge Bases.**
*Dat Quoc Nguyen, Kairit Sirts, Lizhen Qu and Mark Johnson.* NAACL-HLT 2016. [paper](https://arxiv.org/pdf/1606.08140) [code](https://github.com/datquocnguyen/STransE)

1. **GAKE: Graph Aware Knowledge Embedding.**
*Jun Feng, Minlie Huang, Yang Yang, Xiaoyan Zhu.* COLING 2016. [paper](http://yangy.org/works/gake/gake-coling16.pdf) [code](https://github.com/JuneFeng/GAKE)

1. **DKRL: Representation Learning of Knowledge Graphs with Entity Descriptions.**
*Ruobing Xie, Zhiyuan Liu, Jia Jia, Huanbo Luan, Maosong Sun.* AAAI 2016. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12216/12004) [code](https://github.com/thunlp/DKRL)

1. **ProPPR: Learning First-Order Logic Embeddings via Matrix Factorization.**
*William Yang Wang, William W. Cohen.* IJCI 2016. [paper](https://www.cs.ucsb.edu/~william/papers/ijcai2016.pdf)

1. **SSP: Semantic Space Projection for Knowledge Graph Embedding with Text Descriptions.**
*Han Xiao, Minlie Huang, Lian Meng, Xiaoyan Zhu.* AAAI 2017. [paper](http://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/14-XiaoH-14306.pdf)


1. **ProjE: Embedding Projection for Knowledge Graph Completion.**
*Baoxu Shi, Tim Weninger.* AAAI 2017. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14279/13906) [code](https://github.com/bxshi/ProjE)

1. **ANALOGY: Analogical Inference for Multi-relational Embeddings.**
*Hanxiao Liu, Yuexin Wu, Yiming Yang.* ICML 2017. [paper](https://arxiv.org/pdf/1705.02426.pdf) [code](https://github.com/mana-ysh/knowledge-graph-embeddings)

1. **IKRL: Image-embodied Knowledge Representation Learning.**
*Ruobing Xie, Zhiyuan Liu, Tat-Seng Chua, Huan-Bo Luan, Maosong Sun.* IJCAI 2017. [paper](https://www.ijcai.org/proceedings/2017/0438.pdf) [code](https://github.com/xrb92/IKRL)

1. **IPTransE: Iterative Entity Alignment via Joint Knowledge Embeddings.**
*Hao Zhu, Ruobing Xie, Zhiyuan Liu, Maosong Sun.* IJCAI 2017. [paper](https://www.ijcai.org/proceedings/2017/0595.pdf) [code](https://github.com/thunlp/IEAJKE)

1. **On the Equivalence of Holographic and Complex Embeddings for Link Prediction.**
*Katsuhiko Hayashi, Masashi Shimbo.* ACL 2017. [paper](https://aclweb.org/anthology/P/P17/P17-2088.pdf) 

1. **KBGAN: Adversarial Learning for Knowledge Graph Embeddings.**
*Liwei Cai, William Yang Wang.* NAACL-HLT 2018. [paper](https://arxiv.org/pdf/1711.04071.pdf) [code](https://github.com/cai-lw/KBGAN)
