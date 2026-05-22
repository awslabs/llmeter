# Model and Compare Costs

Comparing AI endpoints and models by inference speed alone is, of course, not particularly useful: A holistic evaluation should consider other factors including cost and the various different ways "quality" can be measured too.

Unfortunately, pricing for AI inference is not as simple as common "dollars per input and output token" listings might first appear:
- Many services offer multiple "service tiers", volume discounts, region-specific prices, or other private agreements
- We'd often like to compare as-a-service APIs to other options like pay-per-GPU-hour infrastructure hosting services

To provide the flexibility to model and compare these different pricing structures, LLMeter offers the [`CostModel`](../../reference/callbacks/cost/model/#llmeter.callbacks.cost.model.CostModel) callback and supporting classes in the [`llmeter.callbacks.cost` module](../../reference/callbacks/cost/).

Check out the ["Model Costs" example notebook](https://github.com/awslabs/llmeter/blob/main/examples/Model%20Costs.ipynb) on GitHub for a full walkthrough of how LLMeter can help you attach cost metrics to and compare cost across your test results.
