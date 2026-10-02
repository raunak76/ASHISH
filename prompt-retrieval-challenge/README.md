# PromptMatch: Shipd Eris challenge

Dataset: **Community AI Chat Prompts** (asli data, `fka/prompts.chat`, CC0). Shipd pe status: Ready.

LandslideRisk-48 (`../landslide-challenge/`) chhod diya, kyunki Shipd Discord pe bataya gaya: "It is preferred that you avoid using synthetic datasets for challenges".

## Form ke fields mein kya bharna hai

| Shipd field | Kya daalna hai |
|---|---|
| Dataset | **Community AI Chat Prompts** |
| Difficulty | Hard |
| Compute Tier | CPU (reference solution ~1 min; solvers GPU tier pe bade encoders chala sakte hain) |
| Challenge Title | `PromptMatch: Masked Persona-to-Prompt Retrieval Across Unseen Authors` |
| Problem Description | `problem_description.md` ka poora text |
| Tags | `text` |
| Grade Direction | Maximize |
| Min Score / Max Score | `0` / `1` |
| Grading Script | `grade.py` |
| prepare.py | `prepare.py` |
| Rubrics | `rubrics.md` (abhi platform pe rubrics band hain) |
| Solution | `solution.ipynb` |

## Kya verify kiya gaya hai

- Raw data: 1,915 prompts, 927 contributors, har title alag. 3 duplicate prompts aur 4 duplicate titles hata diye, 1,908 bache.
- `prepare.py`: deterministic hai. Test = 382 prompts, author-disjoint. 196 famous "I want you to act as" prompts sirf train mein hain. Contributor names (47% emails) ki jagah `aNNNN`, aur prompts ke andar ke emails ki jagah `[EMAIL]`.
- Masking: title ke shabd aur unke roop (4+ letter wale shabd pehle 4–5 letters se match) `[MASK]` ban jaate hain. Kisi test prompt mein uske title ka koi shabd nahi bacha.
- `shipd_sim.py`: 18/18 simulated rules pass. Sahi jawab = 1.0 (full, public aur private split pe), sahi jawab rank 2 pe = 0.5, random = 0.005.
- `solution.ipynb`: ~1 minute mein chalta hai (spaCy `en_core_web_md` vectors; na mile toh TF-IDF fallback).

| Method | MRR@10 |
|---|---|
| Random | 0.005 |
| TF-IDF words | 0.021 |
| TF-IDF char 3–5 | 0.033 |
| spaCy vectors, mean cosine (author-held-out validation) | 0.111 |
| **Reference: weighted vector signals** (validation / test) | **0.113 / 0.081** |

Sentence-transformer jaise bade pre-trained encoders yahan test nahi ho paye, kyunki is environment se HuggingFace block hai. Unse kaafi zyada score ki ummeed hai, aur leaderboard ke liye yahi headroom hai.
