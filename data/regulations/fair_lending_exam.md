# Fair Lending Examination Procedures (ECOA / FHA)

## Types of Discrimination
Examiners evaluate three types: overt discrimination, disparate treatment
(treating applicants differently on a prohibited basis), and disparate impact
(a neutral policy with a disproportionate adverse effect that is not justified
by business necessity).

## Disparate Impact Analysis for Models
For statistical and AI models, examiners assess whether model outcomes produce
disparities across prohibited-basis groups, whether the model serves a
legitimate business need, and whether a less discriminatory alternative (LDA)
exists that achieves the same objective.

## Less Discriminatory Alternatives (LDA)
Institutions are increasingly expected to search for LDAs during model
development — e.g., alternative feature sets, model specifications, or
thresholds that reduce disparity while maintaining predictive performance.
Document the search and its results.

## Proxy Methodology
Because protected-class attributes are generally not collected for
non-mortgage credit, disparate-impact testing uses proxy methods (e.g.,
BISG for race/ethnicity) to estimate group membership for analysis.

## Feature Scrutiny
Features that correlate strongly with protected classes (e.g., geography,
certain alternative data) require heightened scrutiny and documentation of
business justification.

## Adverse Action and Explainability
Model-driven adverse actions must yield accurate principal-reason codes.
Complex models (including ML/AI) must be able to generate faithful reason codes;
post-hoc explanation techniques should be validated for fidelity.

## Governance Expectations
Fair-lending risk should be integrated into model risk management: fair-lending
review at development, at validation, and in ongoing monitoring, with results
reported to governance committees.
