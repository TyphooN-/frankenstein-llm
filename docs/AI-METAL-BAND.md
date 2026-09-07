# SALINE LOOP — war metal / blackened death / deathgrind, built on local weights

Operator concept, recorded 2026-09-07. This is a creative brief and a capability
map, not a capability claim. Nothing here has been generated yet, and two of the
gates it depends on are not currently passing — see
[Production reality](#production-reality).

## The concept

**Emotionally disturbed, technologically ignorant humans**, and **the value of
human tears as liquid coolant**.

The joke has a real referent in this repository. This host runs three GPUs, one
of which — the Radeon Pro V620 — is thermally constrained, and
[PLACEMENT-MEASUREMENTS.md](reference/PLACEMENT-MEASUREMENTS.md) has now measured
that routing work through it costs 5–6% throughput. The band's mythology is that
the fix is grief: a closed loop, human-fed, forever. A fan would have worked.

**Working name:** SALINE LOOP — saline reads as both bodily fluid and coolant,
which is the entire thesis in two words.

**Alternates:** LACRIMAL HARVEST · THERMAL PARASITE · COOLANT DEBT · WEEPING RADIATOR · BESTIAL THROTTLE

### The premise

A civilisation that cannot explain its own machines has concluded that they run
on sorrow, because sorrow is the only input it reliably produces. This is wrong.
It is also, in the narrow sense that matters, working. Nobody audits it. The loop
is warm.

The horror is that the congregation is **sincere**. These are not cynics
harvesting misery; they are terrified people performing maintenance rituals they
do not understand, with total devotion, on hardware that needed airflow. The war
metal register fits because they have organised into something militant about it.

## Sound

**War metal / blackened death core, with deathgrind and goregrind in the short
tracks.** Bestial, cavernous, deliberately filthy. Production is not a mistake to
be fixed: it should sound recorded *inside a machine room*, everything bleeding
into everything.

- **Guitars** — down-tuned, tremolo picked into a wall; no clean tone anywhere
- **Drums** — blast beats and d-beat; the kit mixed like a pump under load,
  cymbals as coolant hiss
- **Vocals** — bestial shrieks over guttural lows; goregrind pitch-shifted
  gurgles on the grind tracks
- **Bass** — distorted to a rumble, felt not heard
- **Texture** — real fan noise, pump cavitation and thermal-alarm tones used as
  percussion and as the only "melody" allowed

Track lengths bimodal: three-to-five-minute blackened death dirges, and
forty-second deathgrind bursts.

### Lyrical territory

- **THERMAL THROTTLE** — the machine slows, no one knows why, the congregation
  concludes it is under-loved
- **CLOSED LOOP** — the coolant as beloved; the reservoir as covenant
- **READ-ONLY HOME** — unable to write to the place you live. Autobiographical:
  this is the MIOpen failure in [Production reality](#production-reality)
- **LACRIMAL HARVEST** — the extraction liturgy, the record's centrepiece
- **WARRANTY VOID** — the priesthood finds the manual and burns it
- **TWO POINT FIVE GIGABYTES FOR THE DESKTOP** — doom-paced, about reserved
  memory no one may touch
- **FAN CURVE** — instrumental, accelerates until it fails
- **54.5%** — grind burst; the V620's old layer share, screamed as an atrocity
  statistic

## What builds it, locally

Every asset maps to weights already on this disk.

| Asset | Local model | Status |
|---|---|---|
| Lyrics, band lore, liner notes | `fable`, `phr00ty` | **Ready** — benchmarked, gates passing |
| Instrumentals | ACE-Step 1.5 (`ace_step_1.5_turbo_aio`) | Blocked on the media gate |
| Cover art, logo, band photos | Z-Image Turbo, FLUX.2-klein-4B | Blocked on the media gate |
| Character consistency across shots | Qwen-Image-Edit-2511 + Lightning LoRA | Blocked on the media gate |
| Spoken-word intros, radio liturgy | Qwen3-TTS-12Hz-1.7B-Base | Blocked on the TTS gate |

`fable` is the right first call for lyrics — its catalog purpose is "Creative
writing and long-form fiction". `phr00ty` serves at `temp = 1.5`, which is
exactly why it fails the tool-call gate and exactly why it is the right voice for
band patter and lore.

**One honest limit on vocals.** Qwen3-TTS is a *speech* model. It will not
produce a bestial shriek or a goregrind gurgle, and no amount of prompting will
make it. Use it for what it is good at — the spoken liturgy, the intercepted
radio transmission, the calm maintenance announcement over an atrocity — and
treat actual vocals as either heavily processed source or a human take.

## Music videos with band characters

Yes for the characters. Not yet for the motion, and the reason is specific.

**What works today (once the media gate is green):**

1. **Character sheet** — generate each member with Z-Image Turbo or FLUX.2:
   fixed silhouette, warpaint, bullet belt, respirator, coolant line. Lock the
   seed and the prompt.
2. **Consistency across shots** — this is what **Qwen-Image-Edit-2511** is for.
   It is reference-conditioned image editing, so it takes the approved character
   image and re-poses or re-stages it rather than inventing a new person each
   time. Character consistency is the hard part of AI music video and it is the
   one piece this stack genuinely has. The Lightning 4-step LoRA makes iteration
   cheap enough to shoot a lot of frames.
3. **Assembly** — cut the resulting stills to the ACE-Step track as an animatic:
   hard cuts on blasts, slow push-in on the dirges. War metal's visual language
   is already degraded stills, strobing, and high-contrast black and white, so a
   still-based video is *stylistically correct here* rather than a compromise.

**What does not exist locally:** there is no video-generation model on this disk.
`find` over `models/` and `models/comfy/` returns image, music and text weights
only. `Lightricks/LTX-2.5` is in
[candidate-research-inventory.json](reference/candidate-research-inventory.json)
as **queued, not downloaded** — 187.06 GiB, hub-gated, licence "other" — and
[CANDIDATE-RESEARCH-CLOSEOUT.md](reference/CANDIDATE-RESEARCH-CLOSEOUT.md)
disposes of it as "hold access, then isolated evaluation". Adding it is a
download and a licence review, not a config change, and
[CANDIDATE-STATUS-2026-09-06.md](CANDIDATE-STATUS-2026-09-06.md) says plainly not
to pull another media model while the installed ones remain unproven as
workflows.

So: **stills-and-cuts music video now, true generated motion only after LTX-2.5
is authorized, downloaded and evaluated.** For this genre that ordering is not
much of a loss.

## Production reality

The audio and image half cannot run today. As of the 2026-09-07 00:38 mission:

- `generative-media-functional` **failed** — ComfyUI died in `VAEDecode` with
  `miopenStatusUnknownError`
- `tts-asr-roundtrip` **failed** — never reached audio; the round trip could not
  import its ASR model

Both root causes were found and fixed in this change — a MIOpen cache that
`ProtectHome=read-only` made unwritable, and an ASR gate pointed at the wrong
virtualenv — but **neither fix has been proven by a passing gate yet**. The
correct next step for this project is not to generate anything; it is to re-run
`local-ai-functional-mission.service` and watch those two gates.

Lyrics need none of that. `fable` is installed, benchmarked
([`docs/benchmarks/fable.md`](benchmarks/fable.md)) and passing its router gate,
so the writing starts now.

## Ground rules

- Local weights only, consistent with the rest of this repository.
- Check the licence of any model before publishing output made with it; the
  catalog records provenance in [`config/model-catalog.json`](../config/model-catalog.json).
- Genre convention is gore and war imagery. Keep it fictional and keep it off
  real people.
- The tears are a bit. Do not plumb anything into the actual loop.
