# ADI-Models

To install the requirements:
```
pip install -r requirements.txt
```

Added: Extract frame-level embeddings from ADI-5 models' top intermediate layer


To run:
```
python src/extract_framelevel_embeddings.py --wavlist <wavlist_with_full_path>
```

Output:
A pickled dictonary: where key == wav_id and value == numpy.ndarray

Example:
```
python src/extract_framelevel_embeddings.py --wavlist mgb2_tst_tmp.lst 
```

Output: mgb2_tst_tmp.pickle

Inside mgb2_tst_tmp.lst
```
/export/alt-asr/speech_asr_dataset/mgb2/mgb2_dataset/test2/B8DBA457-2FE6-4A30-B67C-2543E6FAFDAC_spk-0001_seg-0007309___0008131.wav
```
key: B8DBA457-2FE6-4A30-B67C-2543E6FAFDAC_spk-0001_seg-0007309___0008131
value : ndarray, shape : (nframes, 600) 
##TODO
ADD all models (ADI-5 and ADI-17) to get likelihood for dialects

