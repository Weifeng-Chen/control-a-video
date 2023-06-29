# control-a-video
<!-- <img src="basketball.gif" width="256"> -->
Official Implementation of "Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models"

- [Paper Page](https://arxiv.org/abs/2305.13840)

- [Project Page](https://controlavideo.github.io)

# Demo
We support three kinds of control maps at this time. 

|depth control| canny control | hed control | 
|:-:|:-:|:-:|
|<img src="videos/depth_a_bear_walking_through_stars.gif" width="200"><br> |<img src="videos/canny_a_dog_comicbook.gif" width="200"><br> |<img src="videos/hed_a_person_riding_a_horse_jumping_over_an_obstacle_watercolor_style.gif" width="200"><br> |

# Usage
Take depth control for an example:
```
python3 inference.py --prompt $PROMPT --input_video $INPUT_MP4 --control_mode depth 
```
