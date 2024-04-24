# SwissAI

Here, we keep documentation for downloading and processing our datasets. Check out
- [datasets](./datasets/) for instructions on specific datasets
- [configs](./configs/) for default configs (e.g., for video2dataset downloading on clariden in the correct resolutions etc.)
- [processing](./processing/) for instuctions on how to run preprocessing steps
- [overview.ipynb](./dataset_overview.ipynb) for exploring stats on the currently downloaded data


## Dataset Format

We store datasets in the [webdataset format](https://github.com/webdataset/webdataset?tab=readme-ov-file#the-webdataset-format) (`--output_format="webdataset"` when running video2dataset). This means that each shard is stored in a tar file which, for each sample, contains 
- an mp4 file with the video
- an m4a file with the audio (if available)
- a json file with metadata



## TODOs
- [ ] unify datasets
    - make sure each sample can be traced back to where it came from. Add:
        ```
        {
            "original_dataset": "moments_in_time",
            "original_split": "training",
            "original_metadata": {...}
        }
        ```
        (already done for MiT dataset)

    - Unify (resharding etc. if necessary, create index into all samples)

- [ ] Generate missing features
    - [ ] Run WhsiperX over all videos (already supported by v2d) and document workflow
    - [ ] Generate bounding boxes (add [worker](/video2dataset/workers), document workflow)
    - [ ] Generate CLIP features (add [worker](/video2dataset/workers), document workflow)

- [ ] Add Langauge only data
- [ ] Filtering / Deduplication
- [ ] Implement data loader (probably in the model repo)
