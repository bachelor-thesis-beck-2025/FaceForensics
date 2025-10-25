# Fork implementation of 'FaceForensics++: Learning to Detect Manipulated Facial Images' for thesis

## Test detector

Download pretrained model checkpoint and pretrained XceptionNet checkpoint.
Provide test image dataset.

Add path to test images and checkpoints to TEST_pipeline.sh.
```
sh TEST_pipeline.sh
```

This will create a .csv file in the results folder that contains predicted labels, confidence scores and number of faces detected for each image path. 
To upload results to FaceForensics++ benchmark, use benchmark images and run getJson.py on the .csv to get a JSON file in the correct submission format.



From the original repository:

## Citation
If you use the FaceForensics++ data or code please cite:
```
@inproceedings{roessler2019faceforensicspp,
	author = {Andreas R\"ossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
	title = {Face{F}orensics++: Learning to Detect Manipulated Facial Images},
	booktitle= {International Conference on Computer Vision (ICCV)},
	year = {2019}
}

```

