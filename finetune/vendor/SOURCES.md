# Sources

- https://github.com/FIND-Lab/Website-Fingerprinting-Library/tree/537ec8e03c805540b04b95b144fa158ce50aa255
- https://github.com/SJTU-dxw/CountMamba-WF/tree/3d28ccfd6b658f3c04688e9ed6a72ef884e5cd3f

WFlib: seven model files and selected feature functions copied under MIT.
CountMamba: single-tab architecture reimplemented using the local SCSM building blocks; adds token LayerNorm before pooling. No multi-tab / early-stage branch.

Local adjustments: `features.py` retains only the four required functions and fixes the upstream `dir`/`dirs` assertion typo. The model files are unchanged. ARES scheduling is configured by the finetune runner, not copied from upstream training code.
