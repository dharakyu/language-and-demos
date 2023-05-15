# language-and-demos

Repository for Cogsci 2023 paper, "Characterizing tradeoffs between teaching via language and demonstrations with multi-agent systems".

This repository is organized as follows:
- `code` contains all simulation code. The main model training loop is defined in `train.py`. The language and demonstration models are defined in `lang_agent.py` and `demo_agent.py`. 
- `data` contains all run tracking data (`all_experiments.csv`) as well as a sampling of messages for qualitative analysis.
- `analysis` contains the Jupyter notebooks used to analyze the data and create figures.
- `env.yml` lists the required dependencies. Assuming you have conda installed, the environment can be created with `conda env create -f env.yml`

Experiment tracking is configured with Weights and Biases. To log experiments to your account, run `wandb init` in the command line and specify the experiment tracking folder.

To run a one-off experiment, you can run
```
cd code
python train.py [--arguments]
```
See `arguments.py` for a list of arguments and their default values.

The instructions to recreate the figures are as follows:

#### To run the sweep over message capacity (Figure 2):
Note: make sure to run `chmod +x my_script.sh` to make it executable.

```
cd code
./channel_size_lang.sh
./channel_size_pedagogical_demo.sh
./channel_size_random_demo.sh
```

#### To run the sweep over inherent task difficulty (Figure 3A):
```
cd code
./game_difficulty_lang.sh
./game_difficulty_pedagogical_demos.sh
```

#### To run the sweep over agent competence (Figure 3B):
```
cd code
./train_size_lang.sh
./train_size_pedagogical_sampling.sh
```

#### To log example messages/demonstrations (Figure 4 and 5):
```
cd code
./qualitative_analysis_job.sh
```

#### To run additional experiments over task difficulty, also varying the channel size (bonus experiments in Footnote 3):
```
cd code
./game_difficulty_sweep_lang.sh
./game_difficulty_sweep_pedagogical_demos.sh
```
