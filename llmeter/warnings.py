# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warning categories raised by LLMeter.

Defining our own categories (rather than raising bare `UserWarning`s) lets you filter or escalate
LLMeter's warnings precisely, without affecting warnings from other libraries:

```python
import warnings
from llmeter.warnings import LegacyResultFormatWarning

# Silence just this one category
warnings.filterwarnings("ignore", category=LegacyResultFormatWarning)

# Or turn it into an error, e.g. to stop a CI job from drawing conclusions from stale result files
warnings.filterwarnings("error", category=LegacyResultFormatWarning)
```

### Which mechanism does LLMeter use?

Following the guidance in the [Python logging
HOWTO](https://docs.python.org/3/howto/logging.html#when-to-use-logging), LLMeter distinguishes:

* `warnings.warn` - when the issue is *avoidable* and you may want to change your code or how
  you interpret the output. Deprecated arguments and legacy on-disk formats fall here.
* `logger.warning` - when there is nothing you can do about the situation, but the event is still
  worth noting. Unreadable optional files and recovery fallbacks fall here.
"""


class LLMeterWarning(UserWarning):
    """Base class for all warning categories raised by LLMeter.

    Filter on this to control every LLMeter warning at once. Note that *deprecation* warnings
    intentionally use the built-in `DeprecationWarning` instead, so that they behave the way Python
    tooling expects.
    """


class LegacyResultFormatWarning(LLMeterWarning):
    """A result file was written by an older LLMeter version with important semantic differences.

    Raised when loading results saved before a metric's definition changed, so that values are not
    silently compared across incompatible definitions. In most cases, LLMeter does not rewrite the
    loaded values because doing so correctly may not be possible - the warning explains what to
    check.
    """
