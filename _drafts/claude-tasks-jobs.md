---
layout: post
title: Claude is capable of tasks, not jobs
description: A silent bug and rigamarole search
---

# Claude is capable of tasks, not jobs

A few weeks ago I asked Claude to convert nuScenes' ego positions to (lat, lon) so we could compare our reconstructed dashcam track against Google Street View panos. nuScenes stores ego positions as `ego_pose.translation`: an (x, y, z) in metres relative to a per-map anchor. Claude wrote a helper, added a unit test, and the test passed. The values it produced were wrong by hundreds of metres in Boston and tens of metres in Singapore.

I did not catch it for a week. When the evaluation finally broke in ways I couldn't dismiss, I spent three days debugging the wrong things, suspecting nuScenes' own GPS was inaccurate (a known concern with some driving datasets), then suspecting our pano retrieval mechanism and making it better, then suspecting our pose estimator was diverging. Finally, I went back to first principles and asked: *where could a lossy transformation be hiding in the GPS path?* I plotted one Boston GPS on a map. The dot was in some tunnel, the image was taken from a different place. I took the image, asked Claude to read the text on the building and tell me where in boston the image was taken. It gave me a different location than the one I expected, 400 metres away.

## The bug

Claude's helper assumed `ego_pose.translation` was in Web Mercator metres (EPSG:3857) and inverted the projection to recover (lat, lon). nuScenes actually stores it as local ENU metres anchored at the map's southwest corner. Both representations are "metres in a 2D frame," so the function ran without complaint. The values it produced were wrong, by two distinct mechanisms that combine to give the per-city numbers we measured.

**Mechanism 1: `1/cos(latitude)` Mercator scaling.** Web Mercator is conformal but not equal-area; it exaggerates distances by a factor of `1/cos(latitude)` away from the equator. If you misread an ENU offset as a Mercator offset and invert the projection, the recovered point is zoomed about the map origin by that factor. At Boston Seaport (latitude 42.34°) the factor is 1.353, a 35% zoom. At Singapore (latitude 1.3°) it's 1.00025, sub-metre at our scene scales.

**Mechanism 2: spherical Mercator vs. WGS84 ellipsoid.** EPSG:3857 is defined on a sphere; WGS84 (the reference frame for GPS, and the frame nuScenes' map origins are given in) is an ellipsoid. The forward Mercator projection treats the input as if it were on a sphere, so inverting it produces a geodetic (lat, lon) that is off by roughly 0.5% in the north/south direction. This is independent of latitude, so Singapore doesn't escape it: at a few kilometres from the map origin it produces 10–20 m of drift.

Measured per-scene drift was:

| city | scenes | drift range | dominant mechanism |
|---|---|---:|---|
| Singapore | 6 | 6.28 – 18.50 m | sphere/ellipsoid mismatch |
| Boston Seaport | 4 | 192.95 – 551.88 m | Mercator scaling |

## Why the bug survived in the codebase

We were running a few metrics: ATE_Sim3, ATE_Trans, ATE_Rot etc. Most of them align the estimated trajectory to GPS ground truth with a 7-DOF similarity transform (3 translation, 3 rotation, 1 scale) before measuring error. The buggy helper was applied to both the dashcam GPS *and* the GSV pano locations the reconstruction was anchored against, so the entire scene was distorted by the same wrong transformation. Sim3 alignment is exactly the class of operation that absorbs that distortion: small ATE numbers came out of scenes where the underlying ground truth was off by 18 m (Singapore) or 460 m (Boston).

The unit test on the helper was a Mercator → (lat, lon) → Mercator round-trip. It closed because the math was correct *under the assumption that the input was Mercator*.

The Boston scenes did fail end-to-end, just not with a "ground truth is wrong" signature. They failed as `no_panos_queried` and `insufficient_data` — the GSV queries were being issued at the wrong locations, hundreds of metres off the real road, so they returned nothing or nothing useful. We already had a known failure category for genuine GSV-coverage problems with the same symptom, and the bug got booked there.

## Why I didn't catch it for three days

I trusted Claude's helper. It had a clean signature, a docstring, and a passing test. It had been merged. It looked done.

When the evaluation started failing, I debugged the things I had reason to suspect. Driving datasets have GPS noise and was a plausible cause. Our pano database for Boston was thinner than for Singapore and a plausible cause that maybe the testing was done on roads without coverage. Long-sequence drift in the pose estimator was a plausible cause. 

What I did not do for three days was question the GPS-conversion helper, because nothing about it advertised itself as a guess. There was no comment saying "assuming nuScenes uses EPSG:3857." In fact, claude was absolutely certain that it wrote the code correctly.

After three days, I went back to the data path one stage at a time and asked: where could a lossy transformation be hiding? The GPS conversion was the obvious place to look once I was looking. I plotted one frame on a map. The bug was visible immediately.

## Claude does tasks, not jobs

The instruction Claude received was, in effect: *write a function that converts nuScenes map coordinates to (lat, lon).* This is a **task**: it has a typed interface and a name. Claude wrote a function with that interface, picked the projection most commonly associated with "metres → (lat, lon)" on the public web (Web Mercator), wrote a round-trip test that closed, and stopped. Every subgoal of the task was satisfied.

The **job** would have been to notice that "metres → (lat, lon)" is a one-to-many mapping — the answer depends on which projection the input is in — and that the only way to know which projection nuScenes uses is to check the source. There's also the bigger problem that most driving dataset do not state their conventions, making everyone's life harder. A job-holder would have either looked it up and used the right inverse, or written the helper with the assumption flagged in the docstring and a one-line sanity check converting a known landmark.

Claude did neither. It picked a default, produced confident output, clean signature, docstring, passing test and gave no indication of it's guesses. That overconfidence is the active failure here, not the absence of some abstract "ownership." A docstring that said *"assuming nuScenes uses EPSG:3857 — verify before relying on this"* would have caught my eye or even Claude's own guesses.

This is the shape of the LLM-as-developer problem worth being honest about: the model produces confident output that does not distinguish what it knows from what it has guessed. The human downstream also finds it hard to tell either, and pays the cost in debugging time chasing the wrong suspects. Until the model can reliably flag its own assumptions, the human cannot trust the output the way they would trust the output of a colleague who knows what a job is.

## The fix

Replaced the Mercator helpers with `enu_to_ecef(origin_lat, origin_lon, origin_alt, e, n, u)`, which composes the ECEF position of the map origin with the ENU offset rotated into ECEF. Rewrote the round-trip test to start from a known 1 km × 1 km ENU offset, push it through `enu_to_ecef`, recover (lat, lon), and verify the great-circle distance back to the origin using `pyproj.Geod.inv`. The downstream framework works now.
