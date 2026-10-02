# Rubrics (9 criteria)

## 1. REQUIRED · TRAINING
**Criterion:** Validates with folds grouped by `author_id` (no author in both the fitting and the validation part of a fold), and computes MRR@10 by ranking each validation title against all validation-fold prompts.

**Why:** Test authors never appear in train. A random split lets a model match an author's recurring phrasing and overstates retrieval quality on new authors.

## 2. REQUIRED · MODELING
**Criterion:** Uses semantic representations, such as pre-trained word or sentence embeddings, a cross-encoder or a language model, rather than relying only on lexical overlap (TF-IDF/BM25 on words or character n-grams).

**Why:** Title words and their other forms are masked, so lexical retrieval scores about 0.02–0.03 MRR@10. The remaining evidence is related vocabulary and task descriptions.

## 3. REQUIRED · DATA_HANDLING
**Criterion:** Handles very long prompts deliberately, e.g. truncation, chunking with max/mean pooling, or emphasis on the opening where the role is usually defined, instead of passing arbitrarily long text to a model with a fixed input limit and silently losing or crashing on it.

**Why:** Test prompts range from tens of characters to about 50,000. Naive truncation or out-of-memory errors on the longest prompts degrade or break the ranking.

## 4. REQUIRED · DATA_HANDLING
**Criterion:** Does not attempt to undo the masking with external copies of the source dataset (e.g. downloading `fka/prompts.chat` or recalling original prompts by title) and does not use test titles to tune the model.

**Why:** The challenge measures retrieval from meaning on unseen authors. Looking up the original prompts turns it into string matching against leaked labels.

## 5. RECOMMENDED · MODELING
**Criterion:** Uses the 1,526 training pairs to adapt the system, e.g. fine-tuning a bi-encoder with in-batch negatives, learning a combination of similarity signals, or tuning the treatment of `[MASK]` tokens, and shows the gain on author-held-out validation.

**Why:** Off-the-shelf embeddings were not trained on masked prompts or title-to-prompt matching. The training pairs teach that mapping.

## 6. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Treats `[MASK]` explicitly, e.g. removing it before embedding, or using the number and position of masks as features, rather than feeding it as an ordinary word.

**Why:** About 93% of test prompts contain masks. Left as a literal token, `[MASK]` dominates bag-of-words representations and adds noise to embeddings.

## 7. RECOMMENDED · MODELING
**Criterion:** Reports MRR@10 for lexical baselines (word and character TF-IDF) under the same author-held-out validation and shows the final system beats them.

**Why:** The comparison shows the improvement comes from semantic matching, not from residual overlap.

## 8. RECOMMENDED · CODE_QUALITY
**Criterion:** The submission has exactly one row per test `query_id` with up to 10 distinct IDs taken from `test_prompts.csv`, best first.

**Why:** IDs from `train_prompts.csv` or repeated IDs waste ranking positions, and a missing query invalidates the submission.

## 9. RECOMMENDED · MODELING
**Criterion:** Uses the one-to-one structure of the test set, e.g. reranking with a global assignment so that one prompt is not ranked first for many titles, and validates the effect on author-held-out folds.

**Why:** Every test prompt answers exactly one title. Embedding "hubness" makes a few generic prompts top-ranked for many titles, and a global assignment can correct this.
