# Rune Generators

В проекте теперь два готовых варианта:

1. `train_conditional_gan.py` - class-conditional cGAN
2. `train_conditional_diffusion.py` - class-conditional diffusion

Оба тренера:

- читают датасет из `runes/1`, `runes/2`, `runes/3`
- после каждой эпохи сохраняют `4` картинки из фиксированного шума
- пишут превью в `samples/epoch_XXX.png`
- сохраняют чекпоинты в `checkpoints/`

## Без CLI

Все training/sample-сценарии теперь запускаются обычным Python-кодом через конфиги, без `argparse` и без флагов командной строки.

## cGAN

Тренировка:

```python
from pathlib import Path

from train_conditional_gan import GanTrainingConfig, train

summary = train(
    GanTrainingConfig(
        data_dir=Path("runes"),
        epochs=100,
        batch_size=64,
    )
)
print(summary["run_dir"])
```

Сэмплинг:

```python
from pathlib import Path

from sample_conditional_gan import GanSamplingConfig, generate

result = generate(
    GanSamplingConfig(
        checkpoint=Path("artifacts/conditional_gan/<timestamp>/checkpoints/last.pt"),
        class_name="17",
        num_samples=8,
    )
)
print(result["output_dir"])
```

## Diffusion

Тренировка:

```python
from pathlib import Path

from train_conditional_diffusion import DiffusionTrainingConfig, train

summary = train(
    DiffusionTrainingConfig(
        data_dir=Path("runes"),
        epochs=100,
        batch_size=64,
    )
)
print(summary["run_dir"])
```

Сэмплинг:

```python
from pathlib import Path

from sample_conditional_diffusion import DiffusionSamplingConfig, generate

result = generate(
    DiffusionSamplingConfig(
        checkpoint=Path("artifacts/conditional_diffusion/<timestamp>/checkpoints/last.pt"),
        class_name="17",
        num_samples=8,
    )
)
print(result["output_dir"])
```
