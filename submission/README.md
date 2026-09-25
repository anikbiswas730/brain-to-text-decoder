# submission/

`submission.csv` lands here — two columns, `id` and `text`, one row per test trial (1450 in
the reference run), ids in the cache's canonical trial order.

The file is gitignored; upload it by hand.

`decode_llm.py` writes it three times, each overwriting the last:

1. greedy lexicon decode of the first model — exists within minutes of the run starting
2. flashlight 1-best at `screen_lm_weight` — after candidate generation
3. the full gated-fusion prediction (plus selective-GEC rewrites, only if GEC was adopted) — the real submission

That ladder is deliberate: a decoding run that dies in hour six still leaves a valid,
submittable file behind. If the run printed `stages that failed softly: [...]`, the CSV came
from the last stage that succeeded — check that list before submitting.
