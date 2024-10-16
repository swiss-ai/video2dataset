# [HowTo100M](https://github.com/antoine77340/howto100m)
HowTo100M is a dataset of 136M video clips with captions sourced from 1.2M Youtube videos, across 23k activities from domains such as cooking, hand crafting, personal care, gardening or fitness.

## Download the raw metadata
First, run `wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/HowTo100M.zip`

Second, run `python utils/process_howto100m.py -P <RAW ZIP PATH> -O /store/swissai/a08/data/raw/howto100m/v2d -VD` to preprocess the zip file into a CSV compatible with video2dataset, and then download the videos in video2dataset format (if you don't want to download, exclude the `-VD` flag). Note the args for passing in the correct path to the zip and an output dir, as well as optional start/end indices.
This will result in a file `/store/swissai/a08/data/raw/howto100m/v2d/howto100m_v2d.csv` (or, if you pass in start and end indices, `/cluster/work/cotterell/mm_swissai/datasets/howto100m/v2d/howto100m_v2d_<STARTINDEX>_<ENDINDEX>`).

## Download the Videos (from CLI)
While the python script will already download the videos, if you want to download the videos on todi from CLI given the CSV, just run video2dataset with the [default download config for todi](../configs/download_todi.yaml): From the login node, execute the following command, adapting the paths if necessary. Make sure all paths are absolute because the v2d utility will cd around and relative dirs might mess things up a bit.

```
video2dataset --url_list="/store/swissai/a08/data/raw/howto100m/v2d/howto100m_v2d_0_5000.csv" --config="/store/swissai/a08/containers/v2d/video2dataset/swiss_ai/configs/download_todi.yaml" --output_folder="/store/swissai/a08/data/raw/howto100m/v2d" --input_format="csv" --output_format="webdataset" --url_col="video_link" --encode_formats="{'video': 'mp4', 'audio':'m4a'}"
```

OR with `srun`:
```
srun --overlap --jobid=<JOBID> --environment=v2d --container-workdir=$PWD video2dataset --url_list="/store/swissai/a08/data/raw/howto100m/v2d/howto100m_v2d_0_5000.csv" --config="/store/swissai/a08/containers/v2d/video2dataset/swiss_ai/configs/download_todi.yaml" --output_folder="/store/swissai/a08/data/raw/howto100m/v2d" --input_format="csv" --output_format="webdataset" --url_col="video_link" --encode_formats="{'video': 'mp4', 'audio':'m4a'}"
```