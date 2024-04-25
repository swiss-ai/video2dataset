import os
import tarfile
import zipfile
import json
import shutil
import argparse
from tqdm import tqdm
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_zip_path", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()
    raw_zip_path, out_dir = args.raw_zip_path, args.out_dir

    # check if the output directory exists and is empty
    if os.path.exists(out_dir):
        assert len(os.listdir(out_dir)) == 0
    else:
        os.makedirs(out_dir)

    with zipfile.ZipFile(raw_zip_path, "r") as zip_ref:

        print("Analyzing zip file...")

        # read categories
        with zip_ref.open("Moments_in_Time_Raw/moments_categories.txt") as f:
            df_classes = pd.read_csv(f, usecols=[0, 1], names=["name", "id"])
            df_classes.set_index("id", inplace=True)
        print(f"Found {len(df_classes)} classes")

        # read index files
        with zip_ref.open("Moments_in_Time_Raw/trainingSet.csv") as f:
            df_train = pd.read_csv(f, usecols=[0, 1, 2, 3], names=["filename", "label", "Responses 1", "Responses 2"])
        with zip_ref.open("Moments_in_Time_Raw/validationSet.csv") as f:
            df_val = pd.read_csv(f, usecols=[0, 1, 2, 3], names=["filename", "label", "Responses 1", "Responses 2"])
        df_train["path"] = "Moments_in_Time_Raw/training/" + df_train["filename"]
        df_val["path"] = "Moments_in_Time_Raw/validation/" + df_val["filename"]
        df_train["original_split"] = "training"
        df_val["original_split"] = "validation"
        df_samples = pd.concat([df_train, df_val]).reset_index(drop=True)

        # copy over all metadata files
        for video_file in tqdm(zip_ref.namelist()):
            if video_file.endswith("/") or video_file.endswith(".mp4"):
                continue
            print(f"Copying non-mp4 file {video_file}")
            with open(os.path.join(out_dir, os.path.basename(video_file)), "wb") as f:
                f.write(zip_ref.read(video_file))

        # determine the number of digits needed for the shard number etc.
        n_classes, n_videos = len(df_classes), len(df_samples)
        n_digits_class = len(str(n_classes))
        n_digits_video = len(str(n_videos))

        # Treat each directory/class as a shard
        video_id = 0
        n_failures, n_success = 0, 0
        for class_id, class_name in df_classes.iterrows():
            class_name = class_name["name"]
            print(f"Processing class {class_id}/{n_classes-1}: {class_name}...")

            class_key = str(class_id).zfill(n_digits_class)
            class_dir = os.path.join(out_dir, class_key)
            os.makedirs(class_dir, exist_ok=False)
            df_class_metadata = pd.DataFrame(columns=["key", "status", "caption"])

            for local_id, (_, row) in tqdm(enumerate(df_samples[df_samples["label"] == class_name].iterrows())):
                video_file = row["path"]
                video_key = str(video_id).zfill(n_digits_video)

                # extract video
                assert video_file.endswith(".mp4")
                try:
                    data = zip_ref.read(video_file)
                    with open(os.path.join(class_dir, f"{video_key}.mp4"), "wb") as f:
                        f.write(data)
                except KeyError:
                    n_failures += 1
                    print(f"Error extracting {video_file}, (#failures {n_failures} / {n_success})")
                    continue

                # write metadata
                metadata = {
                    "key": video_key,
                    "status": "success",
                    "error_message": None,
                    "caption": class_name,
                }
                dataset_info = {
                    "original_dataset": "moments_in_time",
                    "original_split": row["original_split"],
                    "original_metadata": {
                        "filename": row["filename"],
                        "label": row["label"],
                        "Responses 1": row["Responses 1"],
                        "Responses 2": row["Responses 2"],
                    },
                }
                df_class_metadata.loc[local_id] = metadata
                with open(os.path.join(class_dir, f"{video_key}.json"), "w") as f:
                    f.write(json.dumps({**metadata, **dataset_info}))

                # write caption
                with open(os.path.join(class_dir, f"{video_key}.txt"), "w") as f:
                    f.write(class_name.replace("+", " "))

                video_id += 1
                n_success += 1

            # tar the class directory
            class_tar_fname = str(class_id).zfill(n_digits_class) + ".tar"
            with tarfile.open(os.path.join(out_dir, class_tar_fname), "w") as tar:
                tar.add(class_dir, arcname=class_key)
            shutil.rmtree(class_dir)

            # write metadata
            class_parquet_fname = str(class_id).zfill(n_digits_class) + ".parquet"
            df_class_metadata.to_parquet(os.path.join(out_dir, class_parquet_fname), index=False)

        print(f"Done, #success {n_success}, #failed {n_failures}")


if __name__ == "__main__":
    main()
