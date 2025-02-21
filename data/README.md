# Data process guidance
Here is the guidance of data preparation for training (base and reflow).

## Demo Dataset
Here we provide demo data in `data` folder. The most important files are `metadata_debug.csv`(in `demo_processed_pdb` folder) and `clusters-by-entity-30.txt` (in `metadata` folder), make sure you have properly set the directory of them in `_datasets.yaml`. 

```yaml
scope_dataset:
  ...
  csv_path: ./metadata/scope_metadata.csv # For normal training only
  rectify_csv_path: path/to/rectify_scope_metadata.csv # For reflow training only. Can be ignored in base training
  ...

pdb_dataset:
  ...
  csv_path: path/to/pdb_metadata.csv # For normal training only
  rectify_csv_path: path/to/rectify_pdb_metadata.csv  # For reflow training only. Can be ignored in base training
  cluster_path: ./metadata/clusters-by-entity-30.txt # For normal training only
```

To make a test or debug, we recommend running on demo dataset. 

## Download and process Full PDB Dataset

The following procedure is the same as FrameDiff, but you don't need to clone one more repository.

> WARNING: Downloading PDB can take up to 400GB of space.

To start download, run

```
nohup rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ ./mmCIF > download.log 2>&1 &
```

The process of download can last for hours(~80GB), `nohup` or `tumx` is recommended.(Using nohup above)

After downloading, you should have a directory formatted like this: https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF/

```
00/
01/
02/
..
zz/
```

The folder name indicates protein name inside it.

Then, unzip all files(up to 300GB):

```
cd mmCIF
find . -name '*.gz' -exec gzip -d {} \;
```

Then run the following command to process files.

```
python data/process_pdb_dataset.py --mmcif_dir <mmcif_dir> --write_dir <path_to_write> 
```

See the script for more options. Each mmCIF will be written as a pickle file that we read and process in the data loading pipeline. A `metadata.csv` will be saved that contains the pickle path of each example as well as additional information about each example for faster filtering.

The clustering file is provided in the repo (in `metadata` folder), but can also get from

```
https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt
```

Be sure to correctly config file path as mentioned in `Demo Dataset` section.

## Reflow training
To reflow, you need to inference some data for training on next stage. Run the following code to process generated .pdb files to .pkl that can be used for training. Make sure you have `paired` pdb files in your inference folder. (It should be)

```
python data/rectify_process_pdb_files.py --pdb_dir path/to/inference/dir --write_dir path/to/write/dir
```

In the output folder, there would be a `metadata.csv`. Set `rectify_csv_path` in `_datasets.yaml` to that directory, and data preparation part is done.
