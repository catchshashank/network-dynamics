## 06.03.2026: Danny-Shashank
1) Normalize interaction intensity by time window so comparisons are meaningful.
2) Jaccard similarity results may be driven by sparse activity, not real reshuffling.
   Eg., If a tie has only 1 interaction before and 0 after, the Jaccard similarity becomes 0 by construction.
   So the reshuffling pattern could be statistical artefact rather than behavioral change.
3) Apply interaction thresholds
   Eg., interaction ≥ 2, interaction ≥ 3, and interaction ≥ 5
   This tests robustness of the reshuffling pattern.
4) Focus on top-intensity ties - analyzing only strong ties.
   Eg., Top 10% most active ties or Top X percentile of interaction intensity
   Check whether strong relationships reshuffle, not just weak ties.
5) Screen out low-activity cases - similar to point (2) because several zero-Jaccard values may arise
   simply because there were almost no interactions.
6) Start analysis only after reciprocal tie formation - \tick
7) Include direction of interactions (creator → user interactions) - because if all activity is user → creator, it does not show attention shift by the creator.
8) Run two versions of the analysis -
   - Any user: Focal creator → any user with reciprocal tie
   - Creator-creator only: Focal creator → other creators
   Because creator-creator relationships may represent collaboration networks, not fan interactions.
9) Replicate results under multiple definitions - show that the reshuffling effect is not an artefact.
   - interaction thresholds
   - tie intensity thresholds
   - direction of interaction
   - creator vs user relationships
10) Produce clear “focal graphs”
    - simple focal graphs
    - clear demonstration of reshuffling
    - intuitive figures before complex models
