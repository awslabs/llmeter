# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared prompts for integration tests, each paired with its expected answer.

Kept in one place so that a prompt and the answer asserted against it cannot drift apart, our
multi-provider tests stay consistent between endpoint connectors.

Not named `test_*`, so pytest does not collect it.
"""

# A trivial prompt, for tests that only need a short deterministic completion.
SIMPLE_PROMPT = "What is 15 * 37? Reply with just the number."
SIMPLE_ANSWER = "555"

# A prompt that reliably triggers *adaptive* thinking.
#
# Some models like Claude Opus 4.7 support only `thinking.type: "adaptive"` (`"enabled"` is
# rejected with a 400 error code), which leaves the decision to reason with the *model* and
# requires us to have a reliably reasoning-triggering prompt for testing.
#
# Measured against Opus 4.7 via Bedrock:
#
# * `"What is 15 * 37?"`   -> `thinking_tokens=0`
# * `"What is 127 * 843?"` -> `thinking_tokens=0`
# * ...plus `output_config.effort: "high"` -> still `thinking_tokens=0`
# * this multi-step word problem -> `thinking_tokens=93`
#
# The answer is unambiguous: B + 2.5B + 1.5B = 1290, so B = 258.
REASONING_PROMPT = (
    "A farmer has 3 fields. Field A yields 2.5x Field B. Field C yields 40% less than "
    "Field A. Together they yield 1290 bushels. How many bushels does Field B yield? "
    "Reply with just the number."
)
REASONING_ANSWER = "258"
