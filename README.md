# control-a-video
<!-- <img src="basketball.gif" width="256"> -->
Official Implementation of "Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models"
- [Paper Page](https://arxiv.org/abs/2305.13840)

- [Project Page](https://controlavideo.github.io)

We support three kinds of control maps at this time. 

|depth control| canny control | hed control | 
|:-:|:-:|:-:|
|<img src="videos/depth_a_bear_walking_through_stars.gif" width="200"><br> a bear walking through stars |<img src="videos/canny_a_dog_comicbook.gif" width="200"><br> a dog, comicbook style |<img src="videos/hed_a_person_riding_a_horse_jumping_over_an_obstacle_watercolor_style.gif" width="200"><br> person riding horse, watercolor|


# Env

- torch version: 1.13.1+cu117

```
pip3 install -r requirements.txt
```

# Usage
We provide a demo for quick testing in this repo, simply running:

```
python3 inference.py --prompt "a bear walking through stars, artstation" --input_video bear.mp4 --control_mode depth 
```

More args:
- `--input_video`: path of mp4.
- `--num_sample_frames`: nums of frames to generate. (recommend > 8).
- `--sampling_rate`: skip sampling from the input video.

- `--control_mode`: allows for different control, currently support **`canny`, `depth`, `hed`**. (you need to download the weight of **hed** annotator from [link](https://huggingface.co/wf-genius/controlavideo-hed/resolve/main/hed-network.pth) and put it in work space.)
- `--video_scale`: guidance scale of video consistency, borrows from GEN-1. (don't be too large, 1~2 work well, set 0 to disable it.)
- `--init_noise_thres`: the propoed threshold of residual-based noise init. (range from 0 to 1, larger value leads to more smooth but may introduce artifacts.)

- `--inference_step, --guidance_scale, --height, --width, --prompt`: same as other T2I model.



# Citation
```
@misc{chen2023controlavideo,
        title={Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models}, 
        author={Weifeng Chen and Jie Wu and Pan Xie and Hefeng Wu and Jiashi Li and Xin Xia and Xuefeng Xiao and Liang Lin},
        year={2023},
        eprint={2305.13840},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
```

# Acknowledgement
This repository borrows heavily from [Diffusers](https://github.com/huggingface/diffusers), [ControlNet](https://github.com/lllyasviel/ControlNet), [Tune-A-Video](https://github.com/showlab/Tune-A-Video), thanks for open-sourcing!


# Future Plan
- support lora/dreambooth generation.
- support mask generation.
- optical flow enhancement.

It's also welcomed to contribute any applications based on our models, please propose a PR.