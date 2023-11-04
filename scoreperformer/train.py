""" A minimal training script. """

import argparse

from scoreperformer.experiments import Trainer
from scoreperformer.experiments.callbacks import EpochReproducibilityCallback
from scoreperformer.experiments.components import ExperimentComponents

if __name__ == "__main__":
    parser = argparse.ArgumentParser('training the model')
    parser.add_argument('--config-root', '-r', type=str, default='../recipes')
    parser.add_argument('--config-name', '-n', type=str, default='scoreperformer/base.yaml')

    args = parser.parse_args()

    exp_comps = ExperimentComponents(
        config=args.config_name,
        config_root=args.config_root
    )
    model, train_dataset, eval_dataset, collator, evaluator = exp_comps.init_components()

    trainer = Trainer(
        model=model,
        config=exp_comps.config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
        evaluator=evaluator,
        callbacks=[EpochReproducibilityCallback()]
    )

    trainer.train()
