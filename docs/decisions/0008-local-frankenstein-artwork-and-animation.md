# ADR 0008: Local Frankenstein LLM artwork and audiovisual showcase

- Status: Accepted creative brief; execution blocked until qualification issues are resolved
- Date: 2026-09-09
- Builds on: [ADR 0001](0001-git-tracked-workspace-without-weights.md),
  [ADR 0002](0002-serialized-functional-qualification.md), and
  [ADR 0003](0003-hardware-allocation-and-memory-policy.md)

## Context and prerequisite

The user requests repository artwork that demonstrates the actual local image,
image-to-video and audio capabilities. This is a creative deliverable, not a
replacement for functional qualification or a reason to waive failed checks.
Complete the existing qualification repairs and verify the required workflows
before starting generation. A finished attempt, cached unrelated pass, downloaded
model or successfully queued workflow is not sufficient. Additional media models
needed for the deliverable must be researched and qualified before use.

## Creative brief

Create a graphic with the exact title **Frankenstein LLM** in a brutal death-metal
font/lettering style. Preserve enough legibility to read the repository name.
Underneath the title, depict Frankenstein glowing and crackling with arcing
electricity between various PC components attached to his body.

- Liquid-cooling radiators hang off him on a backpack-like harness.
- Visible tubing connects the radiators, pump and waterblocks into a believable
  cooling loop rather than disconnected decorative hoses.
- Place waterblocks on his heart and brain, with additional PC components and
  cooling details as the composition allows.
- Show the pump at the pelvic region, as mechanical cooling equipment.
- Emphasize dramatic electrical arcs, glowing coolant and recognizably improvised
  PC hardware. Do not silently substitute a generic robot or plain portrait.
- Compose a repository hero image, with a clean title region and a readable
  silhouette at README scale. Save a high-resolution master and a web derivative.

After the still is accepted, animate **that image** into a video with sound.
Preserve the character, title, hardware layout and cooling system. Favor coherent
arcing electricity, pulsing illumination, coolant movement, fan rotation and
restrained camera motion over unstable anatomy, warped lettering or random cuts.
Create an original electrical/mechanical soundscape: crackles, hum, pump and fan
texture, synchronized with salient visible events. Original music is optional;
no copyrighted song or recording is implied by this brief. A short loop is a
sensible first target, not a fixed duration requirement.

## Companion satirical story

The showcase also includes a locally generated satirical story about **TyphooN**,
Frankenstein's creator, and **Frankenstein**. Treat this sourcing detail as a hard
story constraint: **parts came from past builds or shelved inventory except for
the separately sourced 64 GB of RAM, Radeon Pro V620, Heatkiller IV waterblock,
V620 Radeon Pro shroud, new Panaflo fan cables, and Arctic high-pressure 80mm
fans**. The 64 GB describes the newly sourced RAM, not total installed RAM.

User-reported hardware provenance and visual/story details:

Supporting source supplied by the user as proof of the BOINC/distributed-computing
history: https://boincstats.com/stats/-1/user/detail/210227/projectList
Preserve this exact link in the story's provenance notes. Page contents have not
been independently verified (direct extraction failed). Do not treat BOINC project
participation alone as proof of GridCoin earnings, other cryptocurrency holdings,
sale timing, or which physical hardware performed the work.

- Corsair 900D case, battle-scarred from the early Bitcoin mining days, as
  reported by the user. Preserve this worn mining-era history in the story and
  visual design; do not invent specific dates or mining earnings.
- TyphooN is also battle-scarred: in his own account, he sold all his crypto far
  too early. Use this user-invited self-deprecating parallel with the case in the
  satire. Include Dogecoin / GridCoin / Monero alongside Bitcoin in this
  crypto-history backdrop, as requested by the user. Do not infer which coins
  were mined on which components or invent coin-specific transactions.
  Do not invent holdings, sale prices, dates, or hypothetical wealth.

- **36 total fans in the system**. This is the overall count, not 36 of each
  fan type; do not invent the per-model breakdown.

- Panaflo 120MM fans from the early 2000s; new fan cables had to be acquired.
- Gentle Typhoon fans: early-2000s source, as reported by the user; do not invent
  a specific model, quantity, or manufacturing date.
- Arctic high-pressure 80mm fans: purchased this week (relative to this account).
- 2x Swiftech D5 pumps, originating in 2012.
- 2x Swiftech MCR420QP radiators.
- Alphacool nexxxos XT45: a fitting snapped off; the user reports using marine
  JB Weld epoxy and Gorilla Tape to seal it. Depict this as a reported improvised
  repair, not an independently verified leak-tight or safe cooling repair.

Use these specifics in the companion story and relevant artwork details. Do not
invent acquisition dates for the radiators or classify repair supplies as salvaged
without further sourcing information.

Use affectionate technical satire: resurrected hardware, obsessive qualification,
liquid cooling, mismatched generations of PC parts, and a creature confronting
his creator's definition of "spare parts." Keep the creator named TyphooN and
connect the story to the same visual character. This is fictional characterization,
not a factual biography or a benchmark of model intelligence.

Generate the story using a qualified local creative-writing model after the
qualification prerequisite is met. Save the exact prompt, verbatim model output,
model identity and generation settings; retain any human/agent-edited publication
version separately. Check narrative coherence, all recorded sourcing exceptions,
character names and comic quality. A supervisor-written substitute must not be
presented as local-model output. Deliver a readable text/Markdown artifact with
the image, video and provenance manifest.

## Execution decision

Use qualified local runtimes and the existing serialized host-admission path.
Image generation comes first, image-to-video second, and sound generation plus
muxing last. Never overlap memory-heavy media workflows, qualification or local
interactive inference. No cloud generation, paid usage, hardware retuning or
production service changes are authorized by this ADR.

Treat title rendering as its own quality check. If diffusion cannot reproduce
legible death-metal lettering, composite an original or appropriately licensed
local font/lettering layer over the generated artwork; record that step honestly.
Preserve the title as a stable overlay during animation when needed.

Store prompts, negative prompts if used, seeds, source-image hash, workflow graphs,
exact model revisions/quantizations, runtime versions, dimensions, frame rate,
duration, sound settings and executed commands in a reproducibility manifest.
Keep large intermediate media and weights outside Git. Select repository assets
only after visual/audio inspection and size review; public posting, release or
publication requires separate explicit approval.

## Acceptance and evidence

1. Qualification blockers are closed with current valid evidence; required media
   capabilities are independently proven. Do not label an unresolved gate fixed.
2. The saved still visibly contains the title, Frankenstein, PC parts, electrical
   arcs, backpack-mounted radiators, heart/brain waterblocks and pelvic pump.
3. The saved video demonstrably derives from the accepted still and preserves
   visual identity and readable title without major temporal artifacts.
4. The final video has a real, audible sound track. Verify media streams, duration,
   frame dimensions and decoding with local tooling, then inspect representative
   frames and listen to the audio. A mux command returning zero alone is not QA.
5. Deliver absolute paths to the master still, web image, final video with sound,
   and reproducibility manifest. Report actual models, runtimes and execution
   results; never substitute a prompt, plan or fabricated output for the artifacts.

## Current disposition

Brief recorded only. No artwork, animation or soundtrack has been generated by
this decision. Resume after qualification repair, not in parallel with it.
