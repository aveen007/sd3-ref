# SD3 Medium GenEval baseline

| Metric | SD3 paper (~8B) | Native SD3 Medium | Projected T5 | Projected - native |
|---|---:|---:|---:|---:|
| Overall | 0.7400 | **0.71390** | **0.67814** | -0.03576 |
| Single object | 0.9900 | 0.9969 | 1.0000 | +0.0031 |
| Two objects | 0.9400 | 0.8586 | 0.8081 | -0.0505 |
| Counting | 0.7200 | 0.6156 | 0.6562 | +0.0406 |
| Colors | 0.8900 | 0.8723 | 0.8670 | -0.0053 |
| Position | 0.3300 | 0.3350 | 0.2675 | -0.0675 |
| Color attribution | 0.6000 | 0.6050 | 0.4700 | -0.1350 |

Our run used SD3 Medium with 553 prompts, four images per prompt (2,212 images), 1024x1024, 28 steps, CFG 5, shift 3, and seeds 0-3. It achieved 70.52% correct images and 86.62% correct prompts.

The matched projected run used identical settings. It achieved 66.50% correct images, 84.81% correct prompts, and **0.67814** overall. It replaced active T5 content/EOS states while retaining native CLIP-L, CLIP-G, and contextual T5 padding.

The [SD3 paper](https://proceedings.mlr.press/v235/esser24a.html) result is from its larger approximately 8B model, not the released 2B SD3 Medium checkpoint. The [SANA paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/34345e243156da67605d4b63d71c8d98-Paper-Conference.pdf) independently reports 0.62 overall for SD3 Medium. Future projected results must therefore be compared primarily with our matched native baseline of **0.71390**, using identical generation and scoring settings.
