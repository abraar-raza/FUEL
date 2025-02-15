# FUEL
FUEL is a **F**unctional **U**nit-based **E**valuation framework for evaluating the environment impact of **L**LMs.

---
## About
FUEL is implemented based on the open-source LLM inference and serving library, vLLM.  It is designed to evaluate the environmental footprint of LLM serving by integrating performance metrics and carbon emission modeling.

## Getting Started
```bash
pip install -e .
```

### Profiling

FUEL provides data collection modules for both GPU and CPU monitoring. The modules  `gpu_monitor` and `cpu_monitor`, along with the online benchmarking script `benchmark_serving.py`, can be found in the `benchmarks` folder.

To profile your LLM service, refer to the example in `benchmarks/FUEL/test_online_serving.sh`. This script allows you to specify the dataset, input/output lengths, and the queries per second (QPS). Collected data will be persisted in the results directory you specify.

To assess model output quality, refer to the script in `benchmarks/FUEL/quality`.

### Carbon Emission Modeling
Once the profiling metrics are collected, FUEL provides tools to model carbon emissions for each LLM service. You can run the carbon emission modeling using the script `benchmarks/FUEL/carbon_model.py`, which quantifies the environmental impact based on the collected metrics.