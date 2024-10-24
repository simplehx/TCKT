# TCKT: Tree-Based Cross-domain Knowledge Transfer for Next POI Cold-Start Recommendation

The next point of interest (POI) recommendation task recommends POIs to users that they may be interested in next time based on their historical trajectories. This task holds value for both users and businesses. However, it has consistently faced the issue of cold-start caused by sparse user check-in data. Existing research mainly focuses on knowledge transfer among cities within the same data source, but these data are very rare. The abundance of available third-party data presents opportunities to improve cold-start performance, but it is not easy. This third-party data contains numerous entities, such as POIs and users, which have different representations and distributions across different data domains, making knowledge transfer difficult. To address these challenges, we propose the Tree-Based Cross-domain Knowledge Transfer (TCKT) model. First, we construct a multi-granularity Geographical Frequency Tree (GF-Tree), transforming the POI recommendation problem into a path generation problem. Second, we design a pre-training model to mine general user behavior patterns and spatio-temporal features among POIs from large-scale third-party data. Finally, we propose a dual-channel domain adaptation model to facilitate cross-domain knowledge transfer and improve cold-start performance. Experimental results on three public datasets demonstrate that our method outperforms state-of-the-art (SOTA) baseline methods.

## Requirements
* python == 3.10
* lmdb == 1.5.1
* numpy == 1.26.4
* pandas == 2.2.2
* pandarallel == 1.6.5
* pytorch == 2.3.1
* scikit-learn == 1.4.2
* tqdm == 4.66.4
* shapely == 2.0.1


## Datasets
The original dataset can be downloaded from:
* Gowalla: https://snap.stanford.edu/data/loc-gowalla.html
* Brightkite: https://snap.stanford.edu/data/loc-Brightkite.html
* Foursquare: https://sites.google.com/site/yangdingqi/home/foursquare-dataset

The geographic boundary file can be downloaded from:
* https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/

## Usage
* step 1: download dataset to ./raw_data
* step 2: download geographic boundary file to ./us
* step 3: run data_process.py
* step 4: run pretrain.py
* step 5: run finetune.py