# third_party

Source bundled from other projects, kept outside `src/mllmsent/` so it is not
packaged into the wheel and keeps its own licence.

## minigpt4

Upstream: https://github.com/Vision-CAIR/MiniGPT-4 (BSD-3-Clause)

Source only — no weights. It produced the captions in
`data/minigpt4-classify/`, which are published on the Hugging Face dataset repo,
so regenerating them is optional. The eval configs it reads are in
`eval_configs/`.

`import minigpt4` only resolves after this directory is on `sys.path`:

```python
from mllmsent.third_party import enable_minigpt4

enable_minigpt4()
```
