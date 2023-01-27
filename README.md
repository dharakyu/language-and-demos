# signaling-bandits

Repository for Cogsci 2023 submission.

Experiment tracking is configured with Weights and Biases. To log experiments to your account, run `wandb init` in the command line and specify the experiment tracking folder.

#### To run the sweep over message capacity (Figure 2):
```
cd code
./perfect_teacher_lang.sh
./perfect_teacher_pedagogical_demo.sh
./perfect_teacher_random_demo.sh
```

#### To run the sweep over inherent task difficulty (Figure 3):
```
cd code
./perfect_teacher_game_difficulty_lang.sh
./perfect_teacher_game_difficulty_pedagogical_demos.sh
```

#### To run the sweep over agent competence (Figure 4):
```
cd code
./perfect_teacher_train_size_lang.sh
./perfect_teacher_train_size_pedagogical_sampling.sh
```
