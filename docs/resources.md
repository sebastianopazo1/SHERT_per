### Download Required Data

Selective to dowload `data/smplx` and `data/face` from [NJU-Box](https://box.nju.edu.cn/d/a7feb0bf42014f97ae50/) or [Google-Drive](https://drive.google.com/drive/folders/1UokzpgQNGe3q-vdvrWQpD3FreFNy7MmS?usp=sharing).

Dowload SMPLX Model(Male, Female, Neutral) from [SMPL-X](https://smpl-x.is.tue.mpg.de) and put them to `data/models`.

The completed structure should be like:

```
|-- SHERT
    |-- data
        |-- smplx
        |-- masks
        |-- face
        |-- cameras
        |-- models
            |-- smplx
                |-- SMPLX_*.npz
```

### Download Checkpoints

Selective to dowload checkpoints from [NJU-Box](https://box.nju.edu.cn/d/a7feb0bf42014f97ae50/) or [Google-Drive](https://drive.google.com/drive/folders/1UokzpgQNGe3q-vdvrWQpD3FreFNy7MmS?usp=sharing).

Put them to `save/ckpt`. The completed structure should be like:

```
|-- SHERT
    |-- save
        |-- ckpt
            |-- inpaint.pth     # For mesh completion
            |-- refine.pth      # For mesh refinement
            |-- texture_local   # For texture inpainting
            |-- texture_global  # For texture repainting
```


