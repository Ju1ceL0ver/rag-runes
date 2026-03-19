# Rune Generators

В проекте теперь два готовых варианта:

1. `train_conditional_gan.py` - class-conditional cGAN
2. `train_conditional_diffusion.py` - class-conditional diffusion

Оба тренера:

- читают датасет из `runes/1`, `runes/2`, `runes/3`
- после каждой эпохи сохраняют `4` картинки из фиксированного шума
- пишут превью в `samples/epoch_XXX.png`
- сохраняют чекпоинты в `checkpoints/`

## cGAN

```powershell
conda run -n myenv python train_conditional_gan.py --data-dir runes --epochs 100 --batch-size 64
```

Сэмплинг:

```powershell
conda run -n myenv python sample_conditional_gan.py --checkpoint artifacts/conditional_gan/<timestamp>/checkpoints/last.pt --class-name 17 --num-samples 8
```

## Diffusion

```powershell
conda run -n myenv python train_conditional_diffusion.py --data-dir runes --epochs 100 --batch-size 64
```

Сэмплинг:

```powershell
conda run -n myenv python sample_conditional_diffusion.py --checkpoint artifacts/conditional_diffusion/<timestamp>/checkpoints/last.pt --class-name 17 --num-samples 8
```
