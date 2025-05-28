# FUEL
FUEL is a **F**unctional **U**nit-based **E**valuation framework for evaluating the environment impact of **L**LMs.

---
## About
FUEL is implemented based on the open-source LLM inference and serving library, vLLM.  It is designed to evaluate the environmental footprint of LLM serving by integrating performance metrics and carbon emission modeling.

## Getting Started
This project builds on [vLLM](https://github.com/vllm-project/vllm), using version `0.6.1.post2`.  
Please follow their [build-from-source instructions](https://docs.vllm.ai/en/v0.6.1.post2/getting_started/installation.html#build-from-source) to set up dependencies.

To install:

```bash
pip install -e .
```

### Profiling

FUEL provides data collection modules for both GPU and CPU monitoring. The modules  `gpu_monitor` and `cpu_monitor`, along with the online benchmarking script `benchmark_serving.py`, can be found in the `benchmarks` folder.

To profile your LLM service, refer to the example in `benchmarks/FUEL/test_online_serving.sh`. This script allows you to specify the dataset, input/output lengths, and the queries per second (QPS). Collected data will be persisted in the results directory you specify.

To assess model output quality, refer to the script in `benchmarks/FUEL/quality`.

### Carbon Emission Modeling
Once the profiling metrics are collected, FUEL provides tools to model carbon emissions for each LLM service. You can run the carbon emission modeling using the script `benchmarks/FUEL/carbon_model.py`, which quantifies the environmental impact based on the collected metrics.

## Citation
Please cite the following paper if you find FUEL helpful!

```
@article{wu2025unveiling,
  title={Unveiling environmental impacts of large language model serving: A functional unit view},
  author={Wu, Yanran and Hua, Inez and Ding, Yi},
  journal={arXiv preprint arXiv:2502.11256},
  year={2025}
}
```
