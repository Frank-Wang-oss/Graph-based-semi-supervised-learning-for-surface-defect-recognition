# Graph-based-semi-supervised-learning-for-surface-defect-recognition
Pytorch implementation of [A new graph-based semi-supervised method for surface defect classification](https://www.sciencedirect.com/science/article/pii/S0736584520302933?dgcid=coauthor) which is an algorithm for semi-supervised learning in surface defect recognition. (https://github.com/soumith/dcgan.torch).
### Requirements
You will need the following to run the above:
- Pytorch 1.0.1, Torchvision 0.4.1
- Python 3.6.8, Pillow 5.4.1, scikit-learn 0.21.1, numpy 1.16.2
- If you want to train (and don't want to wait for 4 months):
  - A decent GPU
  - All the required NVIDIA software to run TF on a GPU (cuda, etc)
### Dataset
You can access [here](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html), and put the dataset in directory 'NEU-CLS'.
### Citation
If you find the code is useful, please cite our [paper](https://www.sciencedirect.com/science/article/pii/S0736584520302933?dgcid=coauthor)
### Acknowledge
-The project borrowed some code from vgsatorras's [fewshot learning with GNN](https://github.com/vgsatorras/few-shot-gnn)
