# SD3 Medium GenEval baseline

| Metric | SD3 paper (~8B) | Native SD3 Medium | Projected T5 |
|---|---:|---:|---:|
| Overall GenEval | 0.7400 | **0.71390** | **0.67814** |
| Correct images | - | 70.52% (1,560 / 2,212) | 66.50% (1,471 / 2,212) |
| Correct prompts | - | 86.62% (479 / 553) | 84.81% (469 / 553) |
| Single object | 0.9900 | 99.69% (319 / 320) | 100.00% (320 / 320) |
| Two objects | 0.9400 | 85.86% (340 / 396) | 80.81% (320 / 396) |
| Counting | 0.7200 | 61.56% (197 / 320) | 65.62% (210 / 320) |
| Colors | 0.8900 | 87.23% (328 / 376) | 86.70% (326 / 376) |
| Position | 0.3300 | 33.50% (134 / 400) | 26.75% (107 / 400) |
| Color attribution | 0.6000 | 60.50% (242 / 400) | 47.00% (188 / 400) |

Our run used SD3 Medium with 553 prompts, four images per prompt (2,212 images), 1024x1024, 28 steps, CFG 5, shift 3, and seeds 0-3. It achieved 70.52% correct images and 86.62% correct prompts.

The matched projected run used identical settings. It achieved 66.50% correct images, 84.81% correct prompts, and **0.67814** overall. It replaced active T5 content/EOS states while retaining native CLIP-L, CLIP-G, and contextual T5 padding.

The [SD3 paper](https://proceedings.mlr.press/v235/esser24a.html) result is from its larger approximately 8B model, not the released 2B SD3 Medium checkpoint. The [SANA paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/34345e243156da67605d4b63d71c8d98-Paper-Conference.pdf) independently reports 0.62 overall for SD3 Medium. Future projected results must therefore be compared primarily with our matched native baseline of **0.71390**, using identical generation and scoring settings.
