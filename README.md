<p align="center">
  <img src="assets/readme-banner.svg" width="100%" alt="IP Ensemble — First-order influence. Richer insights." />
</p>

### Revisit, Extend, and Enhance Hessian-Free Influence Functions

**Ziao Yang**, Han Yue, Jian Chen, and Hongfu Liu

**Transactions on Machine Learning Research (TMLR), 2026**

**[PDF](https://arxiv.org/pdf/2405.17490)** · **[Paper](https://arxiv.org/abs/2405.17490)** · **[Explore code](#explore-the-code)**

---

Explore first-order influence estimation and IP Ensemble across model utility, fairness, and robustness.

## Explore the code

| Area | Entry point |
| :--- | :--- |
| Synthetic experiments | [`synthetic&adversarial/SEC-4-1.py`](synthetic%26adversarial/SEC-4-1.py) |
| Adversarial experiments | [`run_adversarial.py`](synthetic%26adversarial/run_adversarial.py) · [`run_adversarial.sh`](synthetic%26adversarial/run_adversarial.sh) |
| Fairness and robustness objectives | [`fair_fn.py`](synthetic%26adversarial/fair_fn.py) · [`robust_fn.py`](synthetic%26adversarial/robust_fn.py) |
| NLP resources | [`ResponsibleNLP/`](ResponsibleNLP/) |

## Getting started

```bash
git clone https://github.com/yangziao56/Inner_Product_Ensemble.git
cd Inner_Product_Ensemble
pip install -r requirements.txt
```

Experiment scripts and dataset configurations are organized by task in [`synthetic&adversarial/`](synthetic%26adversarial/).

---

<details>
<summary><strong>Citation</strong></summary>

```bibtex
@article{yang2026innerproductensemble,
  title = {Revisit, Extend, and Enhance Hessian-Free Influence Functions},
  author = {Yang, Ziao and Yue, Han and Chen, Jian and Liu, Hongfu},
  year = {2026},
  journal = {Transactions on Machine Learning Research},
  url = {https://openreview.net/forum?id=ijL2681Tau}
}
```

</details>
