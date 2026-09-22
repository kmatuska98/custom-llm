"""Generate corpus-extension teaching material for negation and spatial_relations,
then verify none of the 48 fixed eval prompts appear verbatim (same check the
notebook runs before training)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def word_tokens(text):
    return re.findall(r"\w+(?:['\u2019]\w+)*|[^\w\s]", text.lower(), flags=re.UNICODE)


def normalized(text):
    return " " + " ".join(word_tokens(text)) + " "


def matching_cases(text, suite):
    content = normalized(text)
    return [c["id"] for c in suite["cases"] if normalized(c["prompt"]) in content]


# ---------------------------------------------------------------- negation ---
names = ["dana", "carlos", "priya", "wes", "tom", "ana", "ben", "ivy",
         "jon", "kira", "mateo", "sofia", "eli", "rosa", "hugo", "nadia"]

item_pairs = [
    ("coffee", "juice"), ("soup", "salad"), ("pasta", "rice"),
    ("cake", "pie"), ("jacket", "sweater"), ("pen", "pencil"),
    ("chair", "stool"), ("map", "guide"), ("bus", "taxi"),
    ("radio", "speaker"),
]

adj_pairs = [
    ("hot", "cold"), ("loud", "quiet"), ("heavy", "light"),
    ("fast", "slow"), ("old", "new"), ("clean", "dirty"),
    ("wet", "dry"), ("full", "empty"), ("bright", "dark"),
    ("sharp", "dull"),
]

nouns = ["room", "shirt", "road", "garden", "kitchen", "hallway",
         "bridge", "field", "cabin", "engine"]

verbs_order = [("order", "ordered"), ("choose", "chose"), ("pick", "picked"), ("request", "requested")]
verbs_finish = [("finish", "finished"), ("return", "returned"), ("pack", "packed"), ("carry", "carried")]
verbs_like = [("like", "likes"), ("prefer", "prefers"), ("want", "wants"), ("need", "needs")]

negation_lines = []
for name in names:
    for a, b in item_pairs:
        for base, past in verbs_order:
            negation_lines.append(f"{name} did not {base} the {a} . {name} {past} the {b} instead .")
for name in names:
    for a, b in item_pairs:
        for base, past in verbs_finish:
            negation_lines.append(f"{name} did not {base} the {a} . {name} {past} the {b} .")
for name in names:
    for a, b in item_pairs:
        for base, third in verbs_like:
            negation_lines.append(f"{name} does not {base} the {a} . {name} {third} the {b} .")
for noun in nouns:
    for a, b in adj_pairs:
        negation_lines.append(f"the {noun} is not {a} . it is {b} .")
        negation_lines.append(f"the {noun} is not {a} . it is {b} . the {noun} looks {b} .")
for a, b in adj_pairs:
    negation_lines.append(f"it was not {a} outside today . it was {b} .")
    negation_lines.append(f"the meeting was not {a} . it was {b} .")
    negation_lines.append(f"the water was not {a} . it was {b} .")

# de-duplicate while preserving order
seen = set()
negation_lines = [l for l in negation_lines if not (l in seen or seen.add(l))]

# --------------------------------------------------------- spatial_relations ---
objects = ["cup", "shelf", "jar", "chair", "table", "bike", "clock",
           "mirror", "stool", "basket", "lantern", "bench", "vase",
           "rug", "ladder", "kettle"]

places = ["cabin", "workshop", "porch", "attic", "pantry", "studio"]

spatial_lines = []
for i, a in enumerate(objects):
    for b in objects[i + 1:]:
        spatial_lines.append(f"the {a} is above the {b} . the {b} is below the {a} .")
        spatial_lines.append(f"the {a} is inside the {b} . the {b} contains the {a} .")
        spatial_lines.append(f"the {a} is left of the {b} . the {b} is right of the {a} .")
        spatial_lines.append(f"the {a} is in front of the {b} . the {b} is behind the {a} .")
        spatial_lines.append(f"the {a} is on top of the {b} . the {b} is under the {a} .")
        spatial_lines.append(f"the {a} is beside the {b} . the {b} is beside the {a} .")

for place in places:
    for obj in objects:
        spatial_lines.append(f"the {place} has the {obj} near the door .")
        spatial_lines.append(f"the {place} keeps the {obj} beside the window .")

seen = set()
spatial_lines = [l for l in spatial_lines if not (l in seen or seen.add(l))]

print(f"negation: {len(negation_lines)} lines")
print(f"spatial:  {len(spatial_lines)} lines")

suite = json.loads((ROOT / "evals/language_evals.json").read_text(encoding="utf-8"))

for label, lines in [("negation", negation_lines), ("spatial_relations", spatial_lines)]:
    text = "\n".join(lines)
    matches = matching_cases(text, suite)
    print(label, "leakage matches:", matches)
    if matches:
        sys.exit(1)

out_dir = ROOT / "corpus_extension"
out_dir.mkdir(exist_ok=True)
(out_dir / "negation.txt").write_text("\n".join(negation_lines) + "\n", encoding="utf-8")
(out_dir / "spatial_relations.txt").write_text("\n".join(spatial_lines) + "\n", encoding="utf-8")
print(f"wrote {out_dir / 'negation.txt'} and {out_dir / 'spatial_relations.txt'}")
print("Copy these into corpus/ only when starting the corpus-extension experiment.")
