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

- `negation.txt`: 2,257 lines using did-not/does-not + correction patterns, across
  16+ names, 10 item pairs (e.g. coffee/juice, jacket/sweater), 10 adjective pairs
  (e.g. hot/cold, wet/dry), 10 general nouns, plus a coverage-fix block (see below).
- `spatial_relations.txt`: 1,656 lines pairing 22 objects with each other across
  6 relations (above/below, inside/contains, left/right, in front of/behind, on top
  of/under, beside), plus places placing objects near a door/window/direction/path.

Both files were checked against `evals/language_evals.json` for exact-prompt leakage
before being written (see `gen_extension_corpus.py` at the repo root) — no eval prompt
appears verbatim in either file. Different names, items, and sentence structures than
the eval's negation/spatial_relations cases were used deliberately, per the assignment's
separation requirement.

### Revision: closing the vocabulary-overlap gap

The first version of this corpus used entirely disjoint nouns/colors/names from the
eval's own negation and spatial-relations cases (e.g. "coffee/juice" instead of
"tea/milk", "cup/shelf" instead of "lamp/desk"). After running it, all 6 of those
eval cases stayed unscorable — not because the model failed to learn the pattern
(samples showed it clearly had), but because the model had never seen the *specific
words* those 6 questions happen to use (red/blue/green/yellow, box, tea/milk/bread,
ava, open/closed, lamp/desk/book/bag/ball, north/south, "to"). The assignment
permits reusing ordinary words as long as the eval's exact sentences aren't
reproduced — avoiding all overlap was overly cautious.

This revision adds those specific words via new sentences, with two things checked
carefully to avoid accidentally reconstructing an eval prompt:
- `lamp`+`desk` and `book`+`bag` are added to the objects list but excluded from
  being paired with *each other* (they still pair with every other object) — pairing
  either combination with this corpus's own "A is above B / B is below A" or
  "A is inside B / B contains A" templates would exactly recreate `lang_41`'s or
  `lang_40`'s prompt.
- Phrasing is kept deliberately different from the eval where words do overlap:
  e.g. this corpus always uses `"the X is not red . it is blue ."` (no trailing
  repeated clause) and `"X did not buy the Y"` (with "the"), never the eval's
  exact `"... . the box is"` / `"ava did not buy tea"` wording.
- `gen_extension_corpus.py`'s leakage check re-verifies the *entire* file text
  (not just individual lines) against all 48 eval prompts, matching exactly how
  the notebook's own `reject_eval_leakage` checks an imported file at import time.

All 6 negation/spatial-relations eval cases now have every prompt and choice word
covered by the extension corpus (verified directly against `tokenization.json`-style
vocabulary extraction, not just visual inspection).

## To run experiment 2

The notebook fetches these files automatically — see the "Optional: fetch my
corpus-extension files from GitHub" cell in `custom_llm.py`/`custom_llm.ipynb`.
Set `FETCH_EXTENSION_FILES = True` in that cell, leave `CORPUS = "classroom"`, and
Run All for a fresh model (don't reuse an earlier run's checkpoint). Manually
copying these files into `corpus/` also works but doesn't survive a Colab runtime
disconnect the way the fetch cell does.
