# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared defaults for the plotting module.

Centralizes template and color management so all plotting functions produce
visually consistent output and respect the same configuration.
"""

from typing import TYPE_CHECKING

from ..utils import DeferredError

if not TYPE_CHECKING:
    try:
        import plotly.io as pio
    except ImportError as e:
        pio = DeferredError(e)
else:
    import plotly.io as pio


#: The default Plotly template used across all llmeter charts.
DEFAULT_TEMPLATE = "plotly_white"


def get_colorway(template: str | None = None) -> list[str]:
    """Get the colorway (qualitative color cycle) from a Plotly template.

    This returns the list of colors that Plotly cycles through for traces
    when no explicit color is provided. By pulling from the template, charts
    automatically adapt if the user switches templates.

    Args:
        template: Template name to pull colors from. Defaults to
            :data:`DEFAULT_TEMPLATE`.

    Returns:
        List of hex color strings.
    """
    template = template or DEFAULT_TEMPLATE
    tmpl = pio.templates[template]
    colorway = tmpl.layout.colorway  # type: ignore[union-attr]
    if colorway:
        return list(colorway)
    # Fallback: Plotly's built-in default colorway
    return list(pio.templates["plotly"].layout.colorway)  # type: ignore[union-attr]
