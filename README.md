## Must-read papers on KRL/KE.
KRL: knowledge representation learning. KE: knowledge embedding.

Contributed by [Shulin Cao](https://github.com/ShulinCao) and [Xu Han](https://github.com/THUCSTHanxu13).

We release [OpenKE](https://github.com/thunlp/openKE), an open source toolkit for KRL/KE. This repository provides a standard KRL/KE training and testing framework. Currently, the implemented models in OpenKE include TransE, TransH, TransR, TransD, RESCAL, DistMult, ComplEx and HolE.

### Survey papers:

1. **Representation Learning: A Review and New Perspectives.**
*Yoshua Bengio, Aaron Courville, and Pascal Vincent.* TPAMI 2013. [paper](https://arxiv.org/pdf/1206.5538)

1. **Knowledge Representation Learning: A Review. (In Chinese)**
*Zhiyuan Liu, Maosong Sun, Yankai Lin, Ruobing Xie.* 计算机研究与发展 2016. [paper](http://crad.ict.ac.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=3099)

1. **A Review of Relational Machine Learning for Knowledge Graphs.**
*Maximilian Nickel, Kevin Murphy, Volker Tresp, Evgeniy Gabrilovich.* Proceedings of the IEEE 2016. [paper](https://arxiv.org/pdf/1503.00759.pdf)

1. **Knowledge Graph Embedding: A Survey of Approaches and Applications.**
*Quan Wang, Zhendong Mao, Bin Wang, Li Guo.* TKDE 2017.  [paper](http://ieeexplore.ieee.org/abstract/document/8047276/)

### Journal and Conference papers:

1. **RESCAL: A Three-Way Model for Collective Learning on Multi-Relational Data.**
*Maximilian Nickel, Volker Tresp, Hans-Peter Kriegel.* ICML 2011. [paper](http://www.icml-2011.org/papers/438_icmlpaper.pdf) [code](https://github.com/thunlp/OpenKE)
    > RESCAL is a tensor factorization approach to knowledge representation learning, which is able to perform collective learning via the latent components of the factorization.

1. **SE: Learning Structured Embeddings of Knowledge Bases.**
*Antoine Bordes, Jason Weston, Ronan Collobert, Yoshua Bengio.* AAAI 2011. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898)
	> SE assumes that the head and tail entities are similar in a relation-dependent subspace, where each relation is represented by two different matrices.

1. **LFM: A Latent Factor Model for Highly Multi-relational Data.**
*Rodolphe Jenatton, Nicolas L. Roux, Antoine Bordes, Guillaume R. Obozinski.* NIPS 2012. [paper](http://papers.nips.cc/paper/4744-a-latent-factor-model-for-highly-multi-relational-data.pdf)
	> LFM is based on a bilinear structure, which captures variouts orders of interaction of the data, and also shares sparse latent factors across different relations.

1. **NTN: Reasoning With Neural Tensor Networks for Knowledge Base Completion.**
*Richard Socher, Danqi Chen, Christopher D. Manning, Andrew Ng.* NIPS 2013. [paper](http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf)
	> NTN is a neural network which allows mediated interaction of entity vectors via a tensor. NTN might be the most expressive model to date, but it is not sufficiently simple and efficient to handle large-scale KGs.

1. **TransE: Translating Embeddings for Modeling Multi-relational Data.**
*Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko.*  NIPS 2013. [paper](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) [code](https://github.com/thunlp/OpenKE)
	> TransE is the first model to introduce translation-based embedding, which interprets relations as the translations operating on entities.

1. **TransH: Knowledge Graph Embedding by Translating on Hyperplanes.**
*Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen.* AAAI 2014. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546) [code](https://github.com/thunlp/OpenkE)
	> To preserve the mapping propertities of 1-N/N-1/N-N relations, TransH inperprets a relation as a translating operation on a hyperplane. In addition, TransH proposes "bern.", a strategy of constructing negative labels.

1. **TransR & CTransR: Learning Entity and Relation Embeddings for Knowledge Graph Completion.**
*Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu.* AAAI 2015. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/) [KB2E](https://github.com/thunlp/KB2E) [OpenKE](https://github.com/thunlp/OpenKE)
	> An entity may have multiple aspects and various relations may focus on different aspects of entites. TransR first projects entities from entity space to corresponding relation space and then builds translations between projected entities.
	CTransR extends TransR by clustering diverse head-tail entity pairs into groups and learning distinct relation vectors for each group, which is the initial exploration for modeling internal correlations within each relation type.

1. **TransD: Knowledge Graph Embedding via Dynamic Mapping Matrix.**
*Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao.* ACL 2015. [paper](http://anthology.aclweb.org/P/P15/P15-1067.pdf) [KB2E](https://github.com/thunlp/KB2E) [OpenKE](https://github.com/thunlp/OpenKE)
	> TransD constructs a dynamic mapping matrix for each entity-relation pair by considering the diversity of entities and relations simultaneously. Compared with TransR/CTransR, TransD has fewer parameters and has no matrix vector multiplication.

1. **TransA: An Adaptive Approach for Knowledge Graph Embedding.**
*Han Xiao, Minlie Huang, Hao Yu, Xiaoyan Zhu.* arXiv 2015. [paper](https://arxiv.org/pdf/1509.05490.pdf)
	> Applying elliptial equipotential hypersurfaces and weighting specific feature dimensions for a relation, TransA can model complex entities and relations.

1. **KG2E: Learning to Represent Knowledge Graphs with Gaussian Embedding.**
*Shizhu He, Kang Liu, Guoliang Ji and Jun Zhao.* CIKM 2015. [paper](https://pdfs.semanticscholar.org/941a/d7796cb67637f88db61e3d37a47ab3a45707.pdf) [code](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/code/cikm15_he_code.zip)
	> Different entities and relations may contain different certainties, which represent the confidence for indicating the semantic when scoring a triple. KG2E represents each entity/relation by a Gaussion distribution, where the mean denotes its position and the covariance presents its certainty.

1. **DistMult: Embedding Entities and Relations for Learning and Inference in Knowledge Bases.**
*Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng.* ICLR 2015. [paper](https://arxiv.org/pdf/1412.6575) [code](https://github.com/thunlp/OpenKE)
	> DistMult is based on the bilinear model where each relation is represented by a diagonal rather than a full matrix. DistMult enjoys the same scalable property as TransE and it achieves superior performance over TransE.

1. **PTransE: Modeling Relation Paths for Representation Learning of Knowledge Bases.**
*Yankai Lin, Zhiyuan Liu, Huanbo Luan, Maosong Sun, Siwei Rao, Song Liu.* EMNLP 2015. [paper](https://arxiv.org/pdf/1506.00379.pdf) [code](https://github.com/thunlp/KB2E)
	> Multi-step relation paths contain rich inference patterns between entities. PTransE considers relation paths as translations between entities and designs an excellent algorithm to measure the reliablity of relation paths. Experiment shows PTransE achieves outstanding improvements on KBC and RE tasks.

1. **RTransE: Composing Relationships with Translations.**
*Alberto García-Durán, Antoine Bordes, Nicolas Usunier.* EMNLP 2015. [paper](http://www.aclweb.org/anthology/D15-1034.pdf)
	> RTransE learns to explicitly model composition of relationships via the addition of their corresponding translations vectors. In addition, the experiments include a new evaluation protocal, in which the model answers questions related to compositions of relations directly.

1. **ManifoldE: From One Point to A Manifold: Knowledge Graph Embedding For Precise Link Prediction.**
*Han Xiao, Minlie Huang and Xiaoyan Zhu.* IJCAI 2016. [paper](https://arxiv.org/pdf/1512.04792.pdf)
	> ManifoldE expands point-wise modeling in the translation-based principle to manifold-wise modeling, thus overcoming the issue of over-strict geometric form and achieving remarkable improvements for precise link prediction.

1. **TransG: A Generative Mixture Model for Knowledge Graph Embedding.**
*Han Xiao, Minlie Huang, Xiaoyan Zhu.* ACL 2016. [paper](http://www.aclweb.org/anthology/P16-1219) [code](https://github.com/BookmanHan/Embedding)
	> A relation in knowledge graph may have different meanings revealed by the associated entity pairs. TransG generates multiple translation components for a relation via a Bayesian non-parametric infinite mixture model.

1. **ComplEx: Complex Embeddings for Simple Link Prediction.**
*Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and Guillaume Bouchard.* ICML 2016. [paper](http://proceedings.mlr.press/v48/trouillon16.pdf) [code](https://github.com/ttrouill/complex) [OpenKE](https://github.com/thunlp/OpenKE)
	> ComplEx extends DistMult by introducing complex-valued embeddings so as to better model asymmetric relations. It is proved that HolE is subsumed by ComplEx as a special case.

1. **ComplEx extension: Knowledge Graph Completion via Complex Tensor Factorization.**
*Théo Trouillon, Christopher R. Dance, Johannes Welbl, Sebastian Riedel, Éric Gaussier, Guillaume Bouchard.* JMLR 2017. [paper](https://arxiv.org/pdf/1702.06879.pdf) [code](https://github.com/ttrouill/complex) [OpenKE](https://github.com/thunlp/OpenKE)

1. **HolE: Holographic Embeddings of Knowledge Graphs.**
*Maximilian Nickel, Lorenzo Rosasco, Tomaso A. Poggio.* AAAI 2016. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12484/11828) [code](https://github.com/mnick/holographic-embeddings) [OpenKE](https://github.com/thunlp/OpenKE)
	> HolE employs circular correlations to create compositional representations. HolE can capture rich interactions but simultaneously remains efficient to compute.

1. **KR-EAR: Knowledge Representation Learning with Entities, Attributes and Relations.**
*Yankai Lin, Zhiyuan Liu, Maosong Sun.* IJCAI 2016. [paper](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/ijcai2016_krear.pdf) [code](https://github.com/thunlp/KR-EAR) 
	> Existing KG-relations can be divided into attributes and relations, which exhibit rather distinct characteristics. KG-EAR is a KR model with entities, attributes and relations, which encodes the correlations between entity descriptions.

1. **TranSparse: Knowledge Graph Completion with Adaptive Sparse Transfer Matrix.**
*Guoliang Ji, Kang Liu, Shizhu He, Jun Zhao.* AAAI 2016. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11982/11693) [code](https://github.com/thunlp/Fast-TransX)
	> The entities and relations in a knowledge graph are heterogeneous and unbalanced. To overcome the heterogeneity, TranSparse uses sparse matrices to model the relations. To deal with the problem of imbalance of relations, each relation has two separate sparse transfer matrices for head and tail entity.

1. **TKRL: Representation Learning of Knowledge Graphs with Hierarchical Types.**
*Ruobing Xie, Zhiyuan Liu, Maosong Sun.* IJCAI 2016. [paper](http://www.thunlp.org/~lzy/publications/ijcai2016_tkrl.pdf) [code](https://github.com/thunlp/TKRL)
	> Entities should have multiple representations in different types. TKRL is the first attempt to capture  the hierarchical types information, which is significant to KRL.

1. **TEKE: Text-Enhanced Representation Learning for Knowledge Graph.**
*Zhigang Wang, Juan-Zi Li.* IJCAI 2016. [paper](https://www.ijcai.org/Proceedings/16/Papers/187.pdf)
	> TEKE incorporates the rich textual content information to expand the semantic structure of the knowledge graph. Thus, each relation is enabled to own different representations for different head and tail entities to better handle 1-N/N-1/N-N relations. TEKE handle the problems of low performance on 1-N/N-1/N-N1 relations and KG sparseness.

1. **STransE: A Novel Embedding Model of Entities and Relationships in Knowledge Bases.**
*Dat Quoc Nguyen, Kairit Sirts, Lizhen Qu and Mark Johnson.* NAACL-HLT 2016. [paper](https://arxiv.org/pdf/1606.08140) [code](https://github.com/datquocnguyen/STransE)
	> STransE is a simple combination of the SE and TransE model, using two projection matrices and one translation vector to represent each relation. STransE produces competitive results on link prediction evaluations.

1. **GAKE: Graph Aware Knowledge Embedding.**
*Jun Feng, Minlie Huang, Yang Yang, Xiaoyan Zhu.* COLING 2016. [paper](http://yangy.org/works/gake/gake-coling16.pdf) [code](https://github.com/JuneFeng/GAKE)
	> Regarding a knowledge base as a directed graph rather than independent triples, GAKE utilizes graph context (neighbor/path/edge context) to learn knowledge representions. Furthermore, GAKE designs an attention mechanism to learn representitive powers of different subjects.

1. **DKRL: Representation Learning of Knowledge Graphs with Entity Descriptions.**
*Ruobing Xie, Zhiyuan Liu, Jia Jia, Huanbo Luan, Maosong Sun.* AAAI 2016. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12216/12004) [code](https://github.com/thunlp/DKRL)
	> DKRL takes advantages of entity descriptions to learn knowledge representations. Outstanding performances under the zero-shot setting indicate that DKRL is capable of building representations for novel entities according to their descriptions. 

1. **ProPPR: Learning First-Order Logic Embeddings via Matrix Factorization.**
*William Yang Wang, William W. Cohen.* IJCAI 2016. [paper](https://www.cs.ucsb.edu/~william/papers/ijcai2016.pdf)
	> ProPPR is the first foraml study to investigate the problem of learning low-dimensional first-order logic embeddings from scratch, while scaling formula embeddings based probabilistic logic reasoning to large knowledge graphs.

1. **SSP: Semantic Space Projection for Knowledge Graph Embedding with Text Descriptions.**
*Han Xiao, Minlie Huang, Lian Meng, Xiaoyan Zhu.* AAAI 2017. [paper](http://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/14-XiaoH-14306.pdf)
	> SSP models the strong correlatons between triples and the textual correlations by performing the embedding process in a sementic improvements against the state-of-the-art baselines.


1. **ProjE: Embedding Projection for Knowledge Graph Completion.**
*Baoxu Shi, Tim Weninger.* AAAI 2017. [paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14279/13906) [code](https://github.com/bxshi/ProjE)
	> ProjE views the KGC task as a ranking problem and projects candidate-entities onto a vector representing a combined embedding of the known parts of an input triple. Besides, ProjE optimizes a ranking loss of the list of candidate-entities collectively. ProjE can be viewed as a simplified version of NTN.

1. **ANALOGY: Analogical Inference for Multi-relational Embeddings.**
*Hanxiao Liu, Yuexin Wu, Yiming Yang.* ICML 2017. [paper](https://arxiv.org/pdf/1705.02426.pdf) [code](https://github.com/mana-ysh/knowledge-graph-embeddings)
	> Analogical inference is of greate use to knowledge base completion. ANALOGY models analogical structure in knowledge embedding. In addition, it is proved that DistMult, HolE and ComplEx are special cases of ANALOGY.

1. **IKRL: Image-embodied Knowledge Representation Learning.**
*Ruobing Xie, Zhiyuan Liu, Tat-Seng Chua, Huan-Bo Luan, Maosong Sun.* IJCAI 2017. [paper](https://www.ijcai.org/proceedings/2017/0438.pdf) [code](https://github.com/xrb92/IKRL)
	> IKRL is the first attemp to combine images with knowledge graphs for KRL. Its promising performances indicate the significance of visual information for KRL.

1. **ITransF: An Interpretable Knowledge Transfer Model for Knowledge Base Completion.**
*Qizhe Xie, Xuezhe Ma, Zihang Dai, Eduard Hovy.* ACL 2017. [paper](https://arxiv.org/pdf/1704.05908.pdf)
	> Equipped with a sparse attention mechanism, ITransF discovers hidden concepts of relations and transfer statistical strength through the sharing of concepts. Moreover, the learned associations between relations and concepts, which are represented by sparse attention vectors, can be interpreted easily.

1. **RUGE: Knowledge Graph Embedding with Iterative Guidance from Soft Rules.**
*Shu Guo, Quan Wang, Lihong Wang, Bin Wang, Li Guo.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.11231.pdf) [code](https://github.com/iieir-km/RUGE)
	> RUGE is the first work that models interactions between embedding learning and logical inference in a principled framework. It enables an embedding model to learn simultaneously from labeled triples, unlabeled triples and soft rules in an iterative manner.

1. **ConMask: Open-World Knowledge Graph Completion.**
*Baoxu Shi, Tim Weninger.* AAAI 2018. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16055/15901)
	> ConMask is a novel open-world Knowledge Graph Completion model that uses relationship-dependent content masking, fully convolutional neural networks, and semantic averaging to extract relationship-dependent embeddings from the textual features of entities and relationships in KGs.

1. **TorusE: Knowledge Graph Embedding on a Lie Group.**
*Takuma Ebisu, Ryutaro Ichise.* AAAI 2018. [paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16227/15885)
	> TorusE defines the principle of TransE on Lie group. A torus, which is one of the compact Lie groups, can be chosen for the embedding space to avoid regularization. TorusE is the first model that embeds objects on other than a real or complex vector space, and this paper is the first to formally discuss the problem of regularization of TransE.

1. **On Multi-Relational Link Prediction with Bilinear Models.**
*Yanjie Wang, Rainer Gemulla, Hui Li.* AAAI 2018. [paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16900/16722) [code](https://dws.informatik.uni-mannheim.de/en/resources/software/tf/)
	> The main goal of this paper is to explore the expressiveness of and the connections between various bilinear models for knowledge graph embedding proposed in the literature. This paper also provides evidence that relation-level ensembles of multiple bilinear models can achieve state-of-the art prediction performance.

1. **Convolutional 2D Knowledge Graph Embeddings.**
*Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, Sebastian Riedel.* AAAI 2018. [paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/17366/15884) [code](https://github.com/TimDettmers/ConvE)
	> ConvE is a multi-layer convolutional network model for link prediction of KGs, and it reports state-of-the-art results for several established datasets. Unlike previous work which has focused on shallow, fast models that can scale to large knowledge graphs, ConvE uses 2D convolution over embeddings and multiple layers of nonlinear features to model KGs.

1. **Accurate Text-Enhanced Knowledge Graph Representation Learning.**
*Bo An, Bo Chen, Xianpei Han, Le Sun.* NAACL-HLT 2018. [paper](http://aclweb.org/anthology/N18-1068) 
	> This paper proposes an accurate text-enhanced knowledge graph representation framework, which can utilize accurate textual information to enhance the knowledge representations of a triple, and can effectively handle the ambiguity of relations and entities through a mutual attention model between relation mentions and entity descriptions.

1. **KBGAN: Adversarial Learning for Knowledge Graph Embeddings.**
*Liwei Cai, William Yang Wang.* NAACL-HLT 2018. [paper](http://aclweb.org/anthology/N18-1133) [code](https://github.com/cai-lw/KBGAN)
	> KBGAN employs adversarial learning to generate useful negative training examples to improve knowledge graph embedding. This framework can be applied to a wide range of KGE models.

1. **ConvKB: A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network.**
*Dai Quoc Nguyen, Tu Dinh Nguyen, Dat Quoc Nguyen, Dinh Phung.* NAACL-HLT 2018. [paper](http://aclweb.org/anthology/N18-2053) [code](https://github.com/daiquocnguyen/ConvKB)
	> ConvKB applies the global relationships among same dimensional entries of the entity and relation embeddings, so that ConvKB generalizes the transitional characteristics in the transition-based embedding models. In addition, ConvKB is evaluated on WN18RR and FB15K237.
	
1. **Modeling Relational Data with Graph Convolutional Networks.**
*Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling.* ESWC 2018. [paper](https://arxiv.org/pdf/1703.06103.pdf) [code1](https://github.com/tkipf/relational-gcn) [code2](https://github.com/MichSchli/RelationPrediction)
	> R-GCN applies Graph Convolutional Networks to relational Knowledge Bases creating a new encoder for the link predicion and entity classification tasks.

1. **Improving Knowledge Graph Embedding Using Simple Constraints.**
*Boyang Ding, Quan Wang, Bin Wang, Li Guo.* ACL 2018. [paper](https://aclweb.org/anthology/P18-1011) [code](https://github.com/iieir-km/ComplEx-NNE_AER)
	> This paper investigates the potential of using very simple constraints to improve KG embedding. It examines non-negativity constraints on entity representations and approximate entailment constraints on relation representations.

1. **Differentiating Concepts and Instances for Knowledge Graph Embedding.**
*Xin Lv, Lei Hou, Juanzi Li, Zhiyuan Liu.* EMNLP 2018. [paper](http://aclweb.org/anthology/DB-1222) [code](https://github.com/davidlvxin/TransC)
	> TransC proposes a novel knowledge graph embedding model by differentiating concepts and instances. Specifically, TransC encodes each concept in knowledge graph as a sphere and each instance as a vector in the same semantic space. This model can also handle the transitivity of isA relations much better than previous models.

1. **SimplE Embedding for Link Prediction in Knowledge Graphs.**
*Seyed Mehran Kazemi, David Poole.* NeurIPS 2018. [paper](https://www.cs.ubc.ca/~poole/papers/Kazemi_Poole_SimplE_NIPS_2018.pdf) [code](https://github.com/Mehran-k/SimplE)
	> SimplE is a simple enhancement of CP (Canonical Polyadic) to allow the two embeddings of each entity to be learned dependently. The complexity of SimplE grows linearly with the size of embeddings. The embeddings learned through SimplE are interpretable, and certain types of background knowledge can be incorporated into these embeddings through weight tying.
	
1. **RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.**
*Zhiqing Sun, Zhi Hong Deng, Jian Yun Nie, Jian Tang.* ICLR 2019. [paper](https://openreview.net/pdf?id=HkgEQnRqYQ) [code](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
	> RotatE infers various relation patterns including: symmetry/antisymmetry, inversion, and composition. Specifically, the RotatE model defines each relation as a rotation from the source entity to the target entity in the complex vector space. 
	
1. **TuckER: Tensor Factorization for Knowledge Graph Completion.**
*Ivana Balazˇevic ́, Carl Allen, Timothy M. Hospedales.* arxiv 2019. [paper](https://arxiv.org/pdf/1901.09590.pdf) [code](https://github.com/ibalazevic/TuckER)
	> TuckER is a relatively simple but powerful linear model based on Tucker decomposition of the binary tensor representation of knowledge graph triples. TuckER is a fully expressive model, deriving the bound on its entity and relation embedding dimensionality for full expressiveness which is several orders of magnitude smaller than the bound of previous models ComplEx and SimplE. Besides, TuckER achieves the state-of-the-art performance.

1. **CrossE: Interaction Embeddings for Prediction and Explanation in Knowledge Graphs.**
*Wen Zhang, Bibek Paudel, Wei Zhang.* WSDM 2019. [paper](https://arxiv.org/pdf/1903.04750.pdf)
	>  CrossE, a novel knowledge graph embedding which explicitly simulates crossover interactions. It not only learns one general embedding for each entity and relation as most previous methods do, but also generates multiple triple specific embeddings for both of them, named interaction embeddings.

1. **Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs.**
*Deepak Nathani, Jatin Chauhan, Charu Sharma, Manohar Kaul.* ACL 2019. [paper](https://arxiv.org/pdf/1906.01195.pdf) [code](https://github.com/deepakn97/relationPrediction)
	> This is a novel attention-based feature embedding model that captures both entity and relation features in any given entity’s neighborhood. This architecture is an encoder-decoder model where the generalized graph attention model and ConvKB play the roles of encoder and decoder respectively.
