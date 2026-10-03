# Component licensing

This repository preserves component-level terms rather than applying a blanket
license to the manuscript, evaluation targets, model predictions and all code.

The multi-step PID code is adapted from the conformalForecast software project,
revision `0fe81e6ff9a80f8c4bb9632b32942d29f3a99127`. Its source is retained under
`docs/references/v4/acmcp/`, including its complete GPL-3 license and author metadata.
The attributed adaptation `src/daci/multistep_pid.py` and `src/daci/acmcp_cached.py` are provided under GPL-3.0 terms.
The license is also available at `licenses/conformalForecast-GPL-3.0.txt`.

Original reference-source notices are retained with their files. The Elsevier article
class and bibliography styles retain their own licensing notices in the paper directory.
No incompatible license is imposed on these components.

SAOCP/FACI reference code retains Salesforce's Apache-2.0 license at
`docs/references/saocp_LICENSE.txt`; `src/daci/reference_rules.py` records its changes.
Earlier PID reference code retains its MIT license at `docs/references/pid_LICENSE.txt`.
These are distinct from the GPL-3 multi-step PID.

The remaining project-specific code and manuscript remain subject to the authors'
copyright; no additional general-purpose open-source license is asserted by this
handoff. Providing a public repository is distinct from granting an unrestricted
license to every component. Data and model terms are described in `DATA_SOURCES.md`.

