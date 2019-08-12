# Global optimization based on generative nerual networks (GLOnet)

## Requirements

We recommend using python3 and a virtual environment

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

A matlab engine for python is needed for EM simulation. Please refer to [MathWorks Pages](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html) for installation.

Path of [RETICOLO](https://www.lp2n.institutoptique.fr/Membres-Services/Responsables-d-equipe/LALANNE-Philippe) should be added in the `main_single.py`

## Train the GLOnet

You can change the parameters by editing `Params.json` in `results` folder. 

If you want to train the network, simply run
```
python main.py 
```

or 

```
python main.py --output_dir results
```

to specify non-default output folder or parameters

## Test model

- test single pair

```
python main.py --restore_from path_of_model --test --wavelength 600 --angle 80
```

- test several pairs randomly

```
python main.py --restore_from path_of_model --test_group --test_num 10
```

- test all pairs in range [600,1200] and [40,80] (step = 10, i.e. 35 pairs)

```
python main.py --restore_from path_of_model --test_group --heatmap
```

## Results

All results will store in output_dir/w\<wavelength\>a\<angle\> folder.
	
	-figures/  (figures of generated devices and loss function curve and others for every 200 iterations)
	
	-model/    (all weights of the generator and discriminator)
	
	-outputs/  (500 generated devices for every combination of wavelength and angle in `.mat` format)
	
	-history.mat
	
	-train.log
