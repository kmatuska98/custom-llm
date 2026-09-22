# Corpus-extension material (do not use yet)

These files are the teaching material for the **second experiment** (corpus extension).
Do **not** copy them into `corpus/` until the starter-corpus experiment (classroom-only)
has already been run and its results saved — `CORPUS = "classroom"` automatically combines
`corpus/`'s contents with the classroom sentences, so adding these now would contaminate
the starter-corpus baseline.

## Chosen categories: `negation` and `spatial_relations`

The classroom corpus (see `classroom_corpus()` in `custom_llm.py`) is built entirely from
8 business/domain word-association templates (customer/service, product/quality,
loan/interest, fruit/juice, car/traffic, software/code, doctor/patient, teacher/lesson).
It contains **zero** negation words ("not", "did not", "does not") and **zero** spatial
relation words (above/below, inside/contains, left/right, in front of/behind, on top
of/under, beside). Both gaps are complete, not partial, so the extension's effect (or
lack of it) should be attributable to missing vocabulary/patterns rather than an
already-partially-covered skill.

They're also a useful contrast: `negation` requires suppressing a nearby mentioned word
and following a correction ("did not buy X, bought Y instead -> answer is Y, not X"),
which is harder for a 2-block/4-head attention model to get exactly right. `spatial_relations`
is more directly associative (learn that "A above B" implies "B below A"), which a tiny
model has a better chance of picking up from repeated co-occurrence alone. Comparing the
two after training should show whether gains differ by pattern difficulty, not just by
"did we add more text."

- `negation.txt`: 2,150 lines using did-not/does-not + correction patterns, across
  16 names, 10 item pairs (e.g. coffee/juice, jacket/sweater), 10 adjective pairs
  (e.g. hot/cold, wet/dry), and 10 general nouns.
- `spatial_relations.txt`: 912 lines pairing each of 16 objects with each other across
  6 relations (above/below, inside/contains, left/right, in front of/behind, on top
  of/under, beside), plus 6 places placing objects near a door/window.

Both files were checked against `evals/language_evals.json` for exact-prompt leakage
before being written (see `gen_extension_corpus.py` at the repo root) — no eval prompt
appears verbatim in either file. Different names, items, and sentence structures than
the eval's negation/spatial_relations cases were used deliberately, per the assignment's
separation requirement.

## To run experiment 2

1. Copy both files into `corpus/` (locally: `cp corpus_extension/*.txt corpus/`; in
   Colab: upload them into `/content/corpus` via the Files sidebar after running
   sections 1-2).
2. Leave `CORPUS = "classroom"` (it will now combine the classroom sentences with these
   two files) and `CORPUS_FOLDER = "corpus"`.
3. Run All again for a fresh model. Do not reuse the starter run's checkpoint.
