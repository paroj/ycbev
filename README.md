# YCB-Ev Dataset

> [!NOTE]
> 2024-09-18: we have released YCB-Ev 1.1 with improved annotations and exteded supplementary programs. Please download the updated annotations from the link below.

The YCB-Ev dataset contains synchronized RGB-D frames and event data that enables evaluating 6DoF object pose estimation algorithms using these modalities.
This dataset provides ground truth 6DoF object poses for the same 21 YCB objects that were used in the YCB-Video (YCB-V) dataset, allowing for cross-dataset algorithm performance evaluation.
The dataset consists of 21 synchronized event and RGB-D sequences, totalling 13,851 frames (7 minutes and 43 seconds of event data). Notably, 12 of these sequences feature the same object arrangement as the YCB-V subset used in the BOP challenge.

Ground truth poses are generated by detecting objects in the RGB-D frames, interpolating the poses to align with the event timestamps, and then transferring them to the event coordinate frame using extrinsic calibration.

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

- new annotations: https://github.com/paroj/ycbev/releases/tag/v1.1
- base dataset: https://github.com/paroj/ycbev/releases/tag/v1
