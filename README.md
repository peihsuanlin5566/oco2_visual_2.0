# OCO2 CO2 observation data visualization tools

This repository places the tools for easily generating CO2 observation map with simple commands. Following the instruction I expected that you are going to be able to generate the figure below. 



![](example-image-co2obs.png)


- Look up at the [mission website](https://www.jpl.nasa.gov/missions/orbiting-carbon-observatory-2-oco-2) for more information regarding OCO2 mission. 


- OCO2 observations can be downloaded at [NASA earth database](https://disc.gsfc.nasa.gov/datasets/OCO2_L2_Lite_FP_10r/summary?keywords=oco-2)

- Description of the OCO2 data (which includes the meaning of all he variables) can be find in the [document provided by NASA](https://docserver.gesdisc.eosdis.nasa.gov/public/project/OCO/OCO2_OCO3_B10_DUG.pdf) 

- After downloading all the data, place them under the same folder and **DO NOT RENAME THE FILE NAME**



## Environment setup  


1. Before setting up the envirnment, you have to ensure you have already with [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages) installed. 
2. clone the repository
```
$ git clone https://github.com/peihsuanlin5566/oco2_visual_2.0.git 
$ cd oco2_visual_2.0
```
3. Run `conda env create -f environment.yml`. 
4. Run `conda activate oco2` to activate the `oco2` environment. 
5. create a `fig` folder for storing the output figures: 
```
$ cd oco2_visual
$ mkdir fig
```
6. create `snowflack_config.json` which contains the information for connecting to the databaase: 
```
$ cat snowflack_config.json 
{   
    "account": "myorganization-myaccount",
    "user" : "testuser1",
    "password" : "0123456",
    "role": "myrole"
}
```


## Usage

Following the instruction in `plotco2map_example.ipynb`. 