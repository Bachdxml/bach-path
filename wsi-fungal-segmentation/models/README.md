Model files for inference live in this folder.

Supported extensions:
- .pth
- .pt
- .ckpt

Notes:
- The desktop app model selector reads this folder.
- If this folder is empty and `checkpoints/best_model.pth` exists,
  the API auto-copies it to `models/default_model.pth`.
- You can keep multiple versions here and choose one in Settings.
