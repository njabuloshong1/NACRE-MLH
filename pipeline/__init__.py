"""Pipeline layer around the tested NACRE-MLH core.

The core (nacre_mlh_cp.py, nacre/, annotators/) is the validated code. This layer reads a run sheet,
drives the core once per dataset, draws the figures and assembles the summary; nothing here decides
a label, so a change in this layer cannot silently change an annotation result.

Two additions were made to the core when it was copied here, both additive:
  * nacre_mlh_cp.py gained --chunk, and sizes it from free memory when it is not given;
  * run_four_bases.R runs each annotator over query chunks, halving and retrying on failure.
A dataset that fits in memory runs in exactly one chunk, which is the original code path, so results
for everything validated so far are unchanged.
"""
