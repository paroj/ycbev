# YCB-Ev Dataset

The YCB-Ev dataset contains synchronized RGB-D frames and event data that enables evaluating 6DoF object pose estimation algorithms using these modalities.
This dataset provides ground truth 6DoF object poses for the same 21 YCB objects that were used in the YCB-Video (YCB-V) dataset, enabling the evaluation of algorithm performance when transferred across datasets.
The dataset consists of 21 synchronized event and RGB-D sequences, amounting to a total of 7:43 minutes of video. Notably, 12 of these sequences feature the same object arrangement as the YCB-V subset used in the BOP challenge.

https://arxiv.org/abs/2309.08482

[![video](https://img.youtube.com/vi/xq_fDXq6Sfw/0.jpg)](https://www.youtube.com/watch?v=xq_fDXq6Sfw)

## Citation

```bibtex
@misc{rojtberg2023ycbev,
      title={YCB-Ev: Event-vision dataset for 6DoF object pose estimation}, 
      author={Pavel Rojtberg and Thomas Pöllabauer},
      year={2023},
      eprint={2309.08482},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Download

https://github.com/paroj/ycbev/releases/tag/v1
