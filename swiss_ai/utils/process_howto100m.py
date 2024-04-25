import pandas as pd
import json
from tqdm import tqdm
import argparse
import zipfile
import os
from video2dataset import video2dataset


def main():
    """
    Starting from the HowTo100M.zip file (download from https://www.di.ens.fr/willow/research/howto100m/), produces a csv in the format expected by video2dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-P", "--raw_zip_path", type=str, default="/cluster/work/cotterell/mm_swissai/datasets/howto100m/HowTo100M.zip"
    )
    parser.add_argument(
        "-O", "--out_dir", type=str, default="/cluster/work/cotterell/mm_swissai/datasets/howto100m/v2d_30000"
    )
    parser.add_argument("-S", "--start_idx", type=int, default=0)
    parser.add_argument("-E", "--end_idx", type=int)
    args = parser.parse_args()
    raw_zip_path, out_dir, start_idx, end_idx = args.raw_zip_path, args.out_dir, args.start_idx, args.end_idx

    # check if the output directory exists and is empty
    if os.path.exists(out_dir):
        if len(os.listdir(out_dir)) != 0:
            raise ValueError(f"Expected out_dir to be empty, instead has {len(os.listdir(out_dir))} files.")
    else:
        os.makedirs(out_dir)

    with zipfile.ZipFile(raw_zip_path, "r") as zip_ref:
        print("Analyzing zip file...")
        # read categories
        with zip_ref.open("caption.json") as f:
            captions_dict = json.load(f)

    flat_data = []
    if end_idx is None:
        end_idx = len(captions_dict)

    for video_id, captions in list(captions_dict.items())[start_idx:end_idx]:
        row = {
            "video_id": video_id,
            "video_link": f"https://youtube.com/watch?v={video_id}",
            "clips": [[a, b] for a, b in zip(captions["start"], captions["end"])],
            "text": captions["text"],
        }
        flat_data.append(row)
    v2d_df = pd.DataFrame(flat_data)

    csv_path = os.path.join(
        out_dir,
        f"howto100m_v2d_{start_idx}_{end_idx}.csv"
        if start_idx != 0 or end_idx != len(captions_dict)
        else "howto100m_v2d.csv",
    )
    v2d_df.to_csv(
        csv_path,
        index=False,
    )

    print("Converting to video2dataset")
    # Convert to v2d format
    video2dataset(
        url_list=csv_path,
        output_folder=out_dir,
        config="swiss_ai/configs/download_clariden.yaml",
        input_format="csv",
        output_format="webdataset",
        url_col="video_link",
        encode_formats=dict(video="mp4", audio="m4a"),
    )


if __name__ == "__main__":
    main()
