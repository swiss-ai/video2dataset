# [HDVILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m)
Page adapted from [dataset_examples/HDVILA.md](/dataset_examples/HDVILA.md).

HDVILA 100M is a dataset of 3.3M high-resolution videos from YouTube, divided into 100M clips.

## Download the metdata
First, run `wget -O hdvila100m.zip https://hdvila.blob.core.windows.net/dataset/hdvila100m.zip?sp=r&st=2022-06-28T03:33:11Z&se=2026-01-01T11:33:11Z&spr=https&sv=2021-06-08&sr=b&sig=VaqQkLFDqKinfkaPNs1jJ1EQIYCB%2FUPYiqFqmjWye6Y%3D` to download the HD VILA 100M metadata. Next, just run `unzip hdvilla100m.zip` in order to unzip the metadata. You should now have an `hdvila100m/` directory.

Next, we need to do some preprocessing to get this metadata formatted into a nice parquet. The following script will take the downloaded metadata `.jsonl` files and create a parquet with all the relevant information.

```python
import pandas as pd
import glob
import json
import os
import time
from datetime import datetime

def time_string_to_seconds(timestamp):
    hh,mm,s = timestamp.split(':')
    ss,ms = s.split('.')
    time = 3600*int(hh) +  60*int(mm) + int(ss) + int(ms)/1000
    return time

def convert_clip_list(clip_list):
    return [[time_string_to_seconds(x) for x in clip] for clip in clip_list]

parquet_dir = "/path/to/my/metadata/dir/"

data = []
for jsonl in sorted(glob.glob(f"{parquet_dir}*.jsonl")):
    path = os.path.join(parquet_dir, jsonl)
    with open(path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            clips = [
                json_obj['clip'][i]['span']
                for i in range(len(json_obj['clip']))
            ]

            out = {
                'video_id': json_obj['video_id'],
                'url': json_obj['url'],
                'clips': clips
            }
            data.append(out)

df = pd.DataFrame(data)
df['clips'] = df['clips'].map(lambda x: convert_clip_list(x))
df.to_parquet("hd_vila.parquet")
```

Once you run this, you should have a file `hd_vila.parquet` with all the relevant metadata.

## Download the Videos
To download the videos on todi, just run video2dataset with the [default download config for todi](../configs/download_todi.yaml): From the login node, execute the following command, adapting the paths if necessary

```
video2dataset --url_list="/store/swissai/a08/data/raw/hdvila/hd_vila.parquet" --config="/store/swissai/a08/containers/v2d/video2dataset/swiss_ai/configs/download_todi.yaml" --output_folder="/store/swissai/a08/data/raw/hdvila/hd_vila_v2d" --input_format="parquet" --output_format="webdataset" --url_col="url" --encode_formats="{'video': 'mp4', 'audio':'m4a'}"
```

This should run at a speed of roughly 500 videos/min or 40 GB/min. For further speedups, consider parallelizing over more nodes.

Also, note that unlike in [dataset_examples/HDVILA.md](/dataset_examples/HDVILA.md) we're not performing cut detection while downloading (at least in the default clariden download config). Cut detection seemed to slow things down considerably and we rather want to perform any such processing on the final combined dataset.
