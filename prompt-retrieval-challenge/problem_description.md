# PromptMatch: Masked Persona-to-Prompt Retrieval Across Unseen Authors

## Overview

Prompt libraries for AI assistants list thousands of community-written prompts under short titles such as "Linux Terminal", "Code Review Expert" or "Travel Guide". Your task is to build a **recommender that, given only a title, retrieves the prompt it belongs to** from a corpus of prompts written by authors the system has never seen.

Each test query is a title. The candidates are all 382 test prompts, and exactly one of them is the prompt written for that title. Submit a ranked list of up to 10 prompt IDs per title.

The data is real: 1,915 community prompts from the public-domain `fka/prompts.chat` collection (CC0), covering software development, education, writing, business, image generation and more.

## Technical focus

Three design choices turn an easy matching exercise into a semantic retrieval problem:

1. **Lexical overlap is removed.** Every word of a title (3+ letters) is replaced by `[MASK]` inside its prompt, including other forms of the same word. Words of 4+ letters are matched on their first 4–5 letters, so "review" also masks "reviewer" and "reviewing", and "code" masks "coder" and "codebase". A prompt for "Linux Terminal" no longer contains "linux" or "terminal", only related evidence ("shell", "commands", "output inside one code block"). Plain TF-IDF drops to MRR@10 ≈ 0.02 (words) and ≈ 0.03 (character n-grams). The match has to come from meaning.
2. **Authors are disjoint.** Train and test are split by author: no test prompt was written by anyone who wrote a training prompt. A system cannot memorise an author's style, recurring phrasing or topics and must generalise across writing styles.
3. **Memorisation is guarded against.** The original "awesome-chatgpt-prompts" list, whose prompts open with "I want you to act as…", is widely mirrored online and likely seen by pre-trained models. Those 196 prompts appear only in train, never in test.

Queries and documents are also highly asymmetric. Titles have a median of 4 words. Test prompts have a median of about 850 characters and the longest about 50,000, and they mix plain instructions, structured JSON/YAML templates and image-generation prompts, and a few are written in French, Turkish and other languages.

## Evaluation

Submissions are scored with **MRR@10** (mean reciprocal rank at 10, higher is better, range 0–1). For each test title, the score is `1 / rank` of the correct prompt among the first 10 distinct prompt IDs listed, or 0 if it is not among them. The final score is the mean over all test titles.

~~~python
def evaluate(ranked, truth, k=10):
    # ranked: {query_id: list of prompt_ids, best first}; truth: {query_id: correct prompt_id}
    total = 0.0
    for q, correct in truth.items():
        top = list(dict.fromkeys(ranked[q]))[:k]      # distinct IDs, listed order
        if correct in top:
            total += 1.0 / (top.index(correct) + 1)
    return total / len(truth)
~~~

For reference, a random ranking scores about 0.005, word TF-IDF about 0.02, character n-gram TF-IDF about 0.03, and averaged pre-trained word vectors about 0.08–0.11.

## Dataset

All files are in `public/`.

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 1,526 | Training queries: `query_id`, `title` and the target `prompt_ids` (the ID of the matching prompt in `train_prompts.csv`) |
| `test.csv` | 382 | Test queries: `query_id`, `title` |
| `train_prompts.csv` | 1,526 | Training prompts (masked), one per training title |
| `test_prompts.csv` | 382 | Test prompts (masked): the candidate pool for every test title |
| `sample_submission.csv` | 382 | Required submission format |

Every test prompt is the answer to exactly one test title. Query and prompt IDs are numbered independently at random, so neither IDs nor row order carry any pairing information.

### Query columns (`train.csv`, `test.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | string | `train_qNNNN` or `test_qNNNN` |
| `title` | string | The prompt's title (role, persona or task name) |
| `prompt_ids` | string | **Target**: ID of the matching prompt. `train.csv` only |

### Prompt columns (`train_prompts.csv`, `test_prompts.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `prompt_id` | string | `train_pNNNN` or `test_pNNNN` |
| `prompt` | string | Prompt text with the title's words masked as `[MASK]`, and e-mail addresses replaced by `[EMAIL]` |
| `type` | string | `TEXT`, `STRUCTURED` or `IMAGE` |
| `for_devs` | bool | Whether the prompt targets developers |
| `author_id` | string | Opaque author ID (`aNNNN`). Train and test authors never overlap |

The source collection's contributor names (many are e-mail addresses) are replaced by opaque author IDs. Exact duplicate prompts and duplicate titles were removed before splitting.

**Pre-trained models are allowed and encouraged** (sentence encoders, word vectors, cross-encoders, small language models), as long as they run in the challenge environment. The `[MASK]` tokens are part of the input. Their count and position are legitimate signals, but no attempt should be made to recover the original text from external copies of the dataset.

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | string | Query ID from `test.csv` |
| `prompt_ids` | string | Up to 10 prompt IDs from `test_prompts.csv`, space-separated, best first |

**Requirements**
- Must contain exactly one row per `query_id` in `test.csv` (382 rows), in any order.
- Include a header row.
- Only the first 10 distinct IDs are scored. Unknown IDs and repeats count as misses.
- Missing or duplicate `query_id` rows make the submission invalid.
