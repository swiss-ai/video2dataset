# [Moments in Time](http://moments.csail.mit.edu)

The Moments in Time dataset contains 1M videos that are ~3s long and labeled with one of 305 categories (e.g., jumping, stirring, selling).

## Download
To download the dataset, you need to request access [here](https://docs.google.com/forms/d/e/1FAIpQLSc0rovlbTCDqJyuJXKLHWtpIX6fiuc1jlAnhT68p86D9NCF9g/viewform) and you will obtain a download link. Then just 
```
curl [link] --output Moments_in_Time.zip
```
This takes ~24h as the server is really slow. There doesn't seem to be a way to speed this up by downloading chunks in parallel, it seems that the server only supports downloading the file starting from the first byte.

## Bring into video2dataset format
We can process the raw data to bring it into video2dataset format using [this python script](../utils/process_mit.py)
```
python process.py --raw_zip_path Moments_in_Time.zip --out_dir out_dir
```

It might take a while and could easily be sped up via some parallelization.
