# Phase 1: Candidate Profiling of LLMs for Multi-Dimensional Evaluation

## Objective

To systematically profile a set of Large Language Models (LLMs) and identify which models are best suited for evaluating specific dimensions (metrics) of text generation quality—namely, Informativeness, Relevance, Fluency, and Coherence—using the human-annotated `newsroom.json` dataset as the validation benchmark.

---

## Background & Motivation

- **Measurement Imbalance:** As highlighted in the MoE-Judge framework and the referenced research, current evaluation paradigms often over-index on technical correctness, neglecting other crucial human-centered dimensions.
- **Symbolic-MoE Inspiration:** The Symbolic-MoE approach profiles LLMs to determine their strengths across various skills or knowledge domains, then routes evaluation tasks accordingly.
- **Goal:** By profiling LLMs on real, multi-dimensional human-annotated data, we can assign each evaluation dimension to the LLM(s) that most closely align with human judgment, thus creating a modular, expert-based judge system.

---

## Step-by-Step Plan

### Step 1: Dataset Preparation

- **Validation Dataset:** Use `newsroom.json` (converted from the Newsroom human evaluation CSV) as the gold standard.
- **Format:** Each instance contains:
  - The generated summary and source article.
  - Human annotation scores for each metric:
    - `mean_human` (average of 3 annotators)
    - `majority_human` (most common score, if any)
    - `individual_human_scores` (list of 3 scores)
  - Metrics: Informativeness, Relevance, Fluency, Coherence.
- **Reference:** See conversion logic in `JUDGE-BENCH/data/newsroom/convert.py`.

### Step 2: Candidate LLMs

- **Source:** The set of LLMs defined in the Symbolic-MoE repo (see `agent_map` in `Symbolic-MoE/agent.py`).
- **Count:** ~18 LLMs, each with a unique model identifier.
- **Assumption:** Each LLM can be prompted to generate or score summaries for the validation set.

### Step 3: Profiling Protocol

- **For Each LLM:**
  1. **Evaluation:** For each instance in `newsroom.json`, have the LLM generate a score for each metric (Informativeness, Relevance, Fluency, Coherence) for the given summary.
  2. **Comparison:** Compare the LLM's score to both the `mean_human` and `majority_human` score for that metric.
  3. **Scoring:**
     - If the LLM's score is within ±1 point of both `mean_human` and `majority_human`, assign **+1** for that metric.
     - If the LLM's score deviates by more than 1 point from both, assign **-1** for that metric.
     - (Optional: If only one is available, use that as the reference.)
  4. **Aggregate:** For each LLM, sum the +1/-1 scores across all instances for each metric to get a profiling score per dimension.

### Step 4: Profiling Output

- **Profile Table:** For each LLM, produce a table or JSON profile:
  - LLM Name
  - Profiling Score for Informativeness
  - Profiling Score for Relevance
  - Profiling Score for Fluency
  - Profiling Score for Coherence
- **Interpretation:** The LLM(s) with the highest profiling score for a given metric are considered "experts" for that dimension.

### Step 5: References and Justification

- **Symbolic-MoE:** The profiling and skill-routing approach is directly inspired by the Symbolic-MoE framework (see `create_profile.py` and `agent.py`).
- **MoE-Judge Paper:** The need for multi-dimensional, modular evaluation is motivated by the measurement imbalance and modularity arguments in the MoE-Judge research abstract and introduction (see `Kushagra Gupta.md`).
- **Dataset Format:** The use of `newsroom.json` and its annotation schema is based on the conversion logic in `convert.py`.

### Step 6: Implementation Notes

- **Automation:** The profiling process should be automated to run each LLM over the validation set and compute the profiling scores.
- **Reproducibility:** Use fixed seeds and document all model versions and parameters.
- **Extensibility:** The profiling protocol can be extended to other datasets or metrics as needed.

---

## References

- [Symbolic-MoE GitHub Repository](link-to-your-repo)
- [MoE-Judge Research Abstract and Introduction](see: Kushagra Gupta.md)
- [Newsroom Dataset and Conversion Script](JUDGE-BENCH/data/newsroom/convert.py)
- [Profiling Logic in Symbolic-MoE](Symbolic-MoE/create_profile.py, Symbolic-MoE/agent.py)

---

## Next Steps
- Proceed to implement the profiling script as described, using the `newsroom.json` validation set and the LLMs in the Symbolic-MoE agent map. 