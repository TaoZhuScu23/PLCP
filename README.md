# Progressive Low-Confidence Pseudolabeling for Semisupervised Node Classification (PLCP)

[![Status: Published](https://img.shields.io/badge/Status-Published-brightgreen)](https://www.sciencedirect.com/science/article/pii/S0925231225018387)
[![Journal: Neurocomputing](https://img.shields.io/badge/Neurocomputing-2025-blue)](https://www.sciencedirect.com/science/article/pii/S0925231225018387)
[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-orange)](https://www.sciencedirect.com/science/article/pii/S0925231225018387)

Official implementation of “Progressive Low-Confidence Pseudolabeling for Semisupervised Node Classification” (Neurocomputing, 2025).  
Authors: Tao Zhu, Hua Mao, Hui Liu, Jie Chen

- Paper: https://www.sciencedirect.com/science/article/pii/S0925231225018387  
- Published: 2025-09-10

---

## Abstract
Graph neural networks (GNNs) have demonstrated remarkable achievements in handling graph-structured data. However, the performance of GNNs is typically limited by the lack of sufficient labeled data, which are time-consuming to obtain in real-world scenarios. Pseudolabeling has been applied to GNNs by augmenting the training set data with unlabeled data. Most pseudolabeling methods on graphs assign pseudolabels to nodes based on high-confidence thresholds. However, nodes near labeled ones generally obtain high confidence scores during training. This results in many similar nodes being assigned pseudolabels, which potentially leads to a distribution shift between the labeled dataset and the augmented dataset. The distribution of the augmented dataset diverges from that of the entire graph data, causing the GNNs to perform poorly on test data. In this paper, we propose a progressive low-confidence pseudolabeling (PLCP) method to progressively leverage the low-confidence data. Specifically, pseudolabels are assigned to nodes within a predefined confidence-based ranking range. To alleviate distribution shift, we keep this range constant to prevent excessive nodes from being assigned pseudolabels. The range is designed to be sufficiently wide to leverage low-confidence nodes. Low-confidence nodes from the range propagate information to their neighbors, which helps the model capture patterns in uncertain regions. To alleviate the impact of noisy pseudolabels, a validation-based reassignment scheme is proposed to assign more reliable pseudolabels. Numerous experiments demonstrate that our proposed PLCP improves the performance of state-of-the-art GNNs on graph datasets.

---

## Getting Started

- Run the code (default settings):
  ```
  python main.py
  ```

---

## Citation

```bibtex
@article{zhu2025progressive,
  title={Progressive low-confidence pseudolabeling for semisupervised node classification},
  author={Zhu, Tao and Mao, Hua and Liu, Hui and Chen, Jie},
  journal={Neurocomputing},
  pages={131166},
  year={2025},
  publisher={Elsevier}
}
```

---

## License and Usage
- This code is released for research purposes.  

---

## Contact
- For questions or issues, please open a GitHub issue in this repository.
