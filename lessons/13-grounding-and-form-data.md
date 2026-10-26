# Stage 13 — Audit grounding and build a form-training dataset

Stage 12 completed two real GRPO updates, but held-out success stayed **0/8**.
One replay showed correctly generated text going into neither intended input,
followed by repeated clicks below Submit. This stage investigates execution
coordinates and prepares relevant demonstrations before spending more GPU time.

This is a **CPU browser job**, with no model loading, optimizer, or new training
checkpoint. Completed run: `stage13-20261019T091807Z-60a38afd`, in
[the Modal app](https://modal.com/apps/kiendepzai3/main/ap-uBeTi5uDFxlKTvp5xPKIVT).
All 120 training episodes passed live replay and offline validation. All 86 CPU
tests passed. Source hashes match the code used for this completed run.

## Run and inspect

```bash
# Local configuration and predefined pools; no browser or PyTorch required.
uv run vlm-multiturn-rl --stage 13
uv run python -m unittest discover -s tests -v

# Replay the completed diagnostic and collect the new training episodes.
uv run modal run modal_app.py --task browser-form-data \
  --audit-run-id stage12-20261019T083719Z-c83d1c0e

# Offline checks, including screenshot hashes, split membership, and targets.
uv run vlm-multiturn-rl --stage 13 --replay artifacts/STAGE13_RUN_ID
```

The remote job uses the existing pinned browser image, two CPUs, 8 GiB host
memory, and a 30-minute timeout. It saves artifacts to the runs volume and
downloads the completed bundle to `artifacts/RUN_ID`. Each launch gets a new
run ID. There is no continuation flag for this data-preparation job.

Open `report.html` for click overlays and `summary.json` for aggregate results.
`episodes.json` contains complete training replays, `images.zip` contains their
screenshots and the audit frames, and `demonstrations.json` contains the exact
current-action training examples. `image-index.json` records PNG hashes and
dimensions. The manifest records source hashes and all artifact checksums.

## 1. Separate a bad prediction from a bad coordinate transform

Playwright uses CSS coordinates relative to the viewport. Our policy emits
original-screenshot coordinates. BrowserGym maps between them using the page's
scale factor; the teacher performs the inverse conversion. See the
[Playwright mouse interface](https://playwright.dev/python/docs/api/class-mouse)
and pinned [BrowserGym coordinate implementation](https://raw.githubusercontent.com/ServiceNow/BrowserGym/v0.14.3/browsergym/core/src/browsergym/core/action/utils.py).

For scale s, the intended contract is

\[
(x_{\rm screenshot},y_{\rm screenshot})
=s(x_{\rm CSS},y_{\rm CSS}).
\]

The audit replays all 16 saved SFT/GRPO evaluation episodes from Stage 12,
without calling either model. It verifies the initial instruction and pixel
hash, then verifies every action's execution/reward/termination result.
Before each action it records real input and Submit boxes. For each click it
captures an actual browser `pointerdown` event and compares its CSS location
with the requested screenshot location divided by s. Differences above one CSS
pixel stop the job; the measured maximum is reported. This tolerance concerns
browser coordinate quantization and has no relation to policy-probability checks.

The standalone report displays a red predicted click over green control boxes.
A click inside a control is not necessarily the correct next action. The audit
therefore reports geometric hits, not a semantic action-accuracy score.

It also records the original screenshot size and the image grids from the
saved model inputs. Processor resizing changes the visual representation;
our action interface still requests original screenshot pixels. The application
does not multiply generated coordinates by a processor resize factor.
This CPU audit cannot establish how accurately the model perceives that image.

The audit's DOM boxes, field values, and focus information live only in
`audit.json` and the diagnostic report. They never enter the training messages
or targets. The old eight evaluation instances are now diagnostic data; they
must not be presented as an untouched final test again.

## 2. Reserve pools before collection

| Pool | Instances | Seeds | Opened in this stage? |
| --- | ---: | --- | --- |
| Clean training | 100 | 50000–50099 | Yes |
| Synthetic recovery training | 20 | 50100–50119 | Yes |
| Development | 16 | 60000–60015 | No |
| Final test | 32 | 70000–70031 | No |

The pool declaration is committed before opening the training instances. New
pools are disjoint from the Stage 12 diagnostic cases. Development is for future
checkpoint selection; final test is reserved until the training procedure is
chosen. Merely checking an oracle on a final-test instance would open it, so
this stage does not do that.

All instances use `multi-orderings`: three fields with shuffled ordering and
different requested values. The benchmark has only six field-order permutations;
100 instances do not create 100 distinct UI structures. The summary reports
order coverage. This studies generalization to new instructions/instances
within a familiar template, not general browser navigation. See the pinned
[task implementation](https://raw.githubusercontent.com/Farama-Foundation/miniwob-plusplus/7fd85d71a4b60325c6585396ec4f48377d049838/miniwob/html/miniwob/multi-orderings.html).

## 3. Demonstrate successful interaction and recovery

The clean teacher uses the same public tools as the policy: click, type, and
press. Its private DOM access chooses control centers and reads current field
values. Every action goes through the real parser and BrowserGym executor.
Every successful episode is then re-executed in a fresh browser. Failure stops
collection rather than silently dropping a difficult seed.

Each recovery episode starts with two deliberately incorrect actions: focus a
chosen input and type `incorrect-training-value`. The reactive teacher then
repairs the form. If a field is nonempty and incorrect, it focuses the field,
uses `press(Control+a)`, and types the replacement. It submits after all three
values are correct. This stays within the unchanged 12-action budget.

These are **synthetic recovery demonstrations**. They are useful coverage of
non-ideal states, but do not constitute DAgger or model-generated failure
collection. DAgger instead labels states encountered by the learner and
aggregates those labels into training; its motivation is that earlier errors
change later observations. See [Ross, Gordon, and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html).

## 4. Keep mistakes as context, not supervised answers

An example stores `case_id`, turn, full public message history, current teacher
action, provenance, and clean/recovery kind. Images remain in chronological
order. The injected incorrect actions are tagged `synthetic_perturbation` and
excluded from targets. They remain in the subsequent teacher action's history,
because that is the state the teacher is correcting.

For future SFT, let P be the multimodal prompt length and a the teacher action
tokens including EOS. Construct

\[
x=[h,a],\qquad
\ell=[\underbrace{-100,\ldots,-100}_{P},a].
\]

The causal model shifts logits and labels for next-token loss. Earlier
assistant mistakes, earlier teacher actions, tool feedback, and image tokens
are context with masked labels. Only the current teacher action is a target.
Do not supervise every assistant message just because it has role `assistant`:
that would teach the deliberate mistakes in recovery histories.

Dataset preparation does not tokenize these examples or claim that token masks
have passed a model check. The next SFT stage must inspect every target span,
EOS, image order, and maximum context length with the actual pinned processor.
That separates CPU data correctness from model-interface validation.

## Common problems and the next experiment

- A successful oracle alone does not independently measure the click transform;
  captured pointer events provide the additional execution check.
- A click can land correctly on the wrong field. Keep geometric hit counts
  separate from instruction-following and complete-episode success.
- Low teacher-forced loss does not establish recovery under free-running actions.
  Evaluate complete rollouts on development instances after training.
- Synthetic mistakes cover only the injected error pattern. Add oracle
  corrections on learner-generated **training** states if errors persist.
- Counting optimizer updates without counting examples is misleading. The new
  dataset is much larger than the original 18-action demonstration set; choose
  SFT duration using dataset passes and development outcomes.

The next GPU experiment should initialize from the validated Stage 9 adapter,
train on this dataset with current-assistant-only masks, and compare before/after
on the declared development pool. Keep the final-test pool closed. GRPO follows
once sampled training rollouts show successes across more than two instances.

## Measured results

| Measurement | Result |
| --- | --- |
| Saved model episodes audited | 16: eight SFT and eight GRPO |
| Actual model clicks checked | 132 |
| Clicks inside any form control | 33 |
| Clicks outside all form controls | **99 / 132 (75%)** |
| Maximum actual/requested pointer discrepancy | **0.000007121 CSS pixels** |
| Clean teacher success and fresh-browser replay | **100/100**, seven actions each |
| Recovery teacher success and fresh-browser replay | **20/20**, seven nine-action and thirteen ten-action episodes |
| Current teacher-action targets | **853**: 700 clean, 153 recovery |
| Injected mistakes excluded from targets | **40** |
| Recovery `Control+a` actions | 20 |
| Field-order coverage | All six permutations; 11–23 clean instances per ordering |
| Development / final instances opened | **0 / 0** |
| Optimizer updates | **0** |
| CPU function runtime | **1338.3 seconds / 22.3 minutes** |
| Downloaded bundle before local convenience preview | About 19 MiB |

The execution discrepancy is negligible on these recorded clicks. These
measurements support investigating the policy's chosen locations rather than
applying an arbitrary global coordinate correction. They do not isolate the
model's perception, coordinate prediction, or planning as the sole cause.
Even a geometric hit can select the wrong control.

Of the model clicks, SFT missed every control on 47/63 and GRPO on 52/69.
Private focus inspection also found 11 SFT typing actions and nine GRPO typing
actions with no input focused. These are diagnostics of the previously recorded
failures, not new evaluation results or evidence of improvement from this data.

The first recovery preview visibly changes an incorrect genre value into the
requested genre and fills the other fields correctly. No model has been trained
on these examples yet; 120/120 is the **teacher's** result.

Open the completed [click-overlay report](../artifacts/stage13-20261019T091807Z-60a38afd/report.html),
[local clean/recovery preview](../artifacts/stage13-20261019T091807Z-60a38afd/training-preview.html),
or [dataset summary](../artifacts/stage13-20261019T091807Z-60a38afd/summary.json).
The convenience training preview was generated locally from the verified bundle;
the remote report and core data are covered by its manifest.

## Exercise

Find the first correction target in one recovery episode. Identify the two
earlier mistakes in its history. Explain why those mistakes remain visible to
attention while receiving no supervised labels, and why teacher DOM access
does not belong in the policy observation.
