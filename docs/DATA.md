# Dataset facts

**Generated** 2026-09-01 by `scripts/dataset_stats.py`.
**Blocks** 44,429,969 - 50,749,122. **Decoded events** 224,050.

Do not hand-edit. Do not quote ACP figures anywhere - README, spec, Devpost,
video, research post - that did not come from this file. Re-run the script
after every scanner catch-up.

## Funnel

| | count |
|---|---|
| jobs created | 75,340 |
| jobs funded | 18,367 (24.38% of created) |
| funded and completed | 14,597 |
| died after funding | 3,770 |
| **base rate** P(completion \| funded) | **0.7947** |

## Failure modes

Among the 3,770 funded jobs that never completed. These overlap - a
rejection typically triggers a refund - and they are explanatory fields on a
failure episode, never competing labels.

| terminal event | count |
|---|---|
| JobRejected | 650 |
| JobExpired | 3,110 |
| Refunded | 3,382 |

Across **all** jobs, funded or not, there were 1,667
rejections: 894 pressed by the provider,
762 by the client,
11 by neither.

## Agents

| | count |
|---|---|
| distinct agents (client or provider) | 718 |
| with 5+ jobs | 440 |
| with 20+ jobs | 213 |
| providers with 1+ funded job | 271 |

The two busiest clients created 55,004 of 75,340 jobs
(73.01%), so the marketplace is heavily
concentrated on the demand side.

## Per-provider completion, funded jobs only

Providers with at least 150 funded jobs.

| provider | funded | completed | rate |
|---|---|---|---|
| `0xdf5e15c898c869340135f39040bdf451a93f9dd8` | 3,039 | 3,033 | 1.00 |
| `0x44cc25d55a4291b92f52062ba023ca1f14206664` | 2,022 | 2,014 | 1.00 |
| `0xd6a5093213d0e940a887ee5327c60af7e53b0261` | 1,982 | 1,982 | 1.00 |
| `0xe09f40114af6c78788a8003da127c49c56158584` | 1,941 | 1,936 | 1.00 |
| `0x7457b799121c9b8c51298d08f1c19f0186648c90` | 1,647 | 88 | 0.05 |
| `0x3899523d94289a94afa3b33745bc95c125483041` | 1,027 | 773 | 0.75 |
| `0xea0f80aac331a0aee486ee67d61f9f4ad7085ee8` | 694 | 469 | 0.68 |
| `0xccf57d593aa2cf8f8329e5e8353f69f2b21e34e9` | 431 | 402 | 0.93 |
| `0xfcf0b00f8352dd193a671e40940c9a32396adb49` | 402 | 383 | 0.95 |
| `0xd1baa920868a66e213383be798a415ef45e8cefe` | 399 | 98 | 0.25 |
| `0x4cdb0d2fd8755a7c3924b025cd953d665d023c2b` | 392 | 368 | 0.94 |
| `0x9a09f5c61079712be9d85a32bdbbd9d1abb5298f` | 253 | 132 | 0.52 |
| `0x2d71d98345cf06b4f26294465406df707697c4ef` | 225 | 184 | 0.82 |
| `0xa41cde7183313f521b565fccf299fda717aed4ee` | 192 | 113 | 0.59 |
| `0x94c037393ab0263194dcfd8d04a2176d6a80e385` | 190 | 20 | 0.11 |
| `0x3cc44b0ab1735cc3df1b2194e2b4ac0ab099b25a` | 178 | 169 | 0.95 |
| `0xa9667116b4f4e9f1bae85f93a21b4b8ea45de98f` | 168 | 110 | 0.65 |

## Evaluator independence

Over 14,644 completed jobs.

| evaluator is | count | share |
|---|---|---|
| the client who created the job | 8,659 | 59.13% |
| absent | 5,967 | 40.75% |
| the provider | 0 | 0.00% |
| **a genuine third party** | **18** | **0.12%** |

Of 7,356 evaluator fee payments,
7,338
(99.76%) went to the
client who created the job.

ACP defines an evaluator role for independent verification. In practice it is
almost never independent.
