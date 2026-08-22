import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from finetune import trainer as trainer_module


class _SizedLoader:
    def __init__(self, length, prepared=False):
        self.length = length
        self.prepared = prepared

    def __len__(self):
        return self.length


class _FakeAccelerator:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.state = SimpleNamespace(deepspeed_plugin=None)
        self.prepare_data_loader_calls = 0

    def prepare_data_loader(self, data_loader):
        if data_loader.prepared:
            return data_loader
        self.prepare_data_loader_calls += 1
        return _SizedLoader(math.ceil(len(data_loader) / self.num_processes), prepared=True)

    def prepare(self, transformer, optimizer, data_loader, scheduler):
        data_loader = self.prepare_data_loader(data_loader)
        return transformer, optimizer, data_loader, scheduler


class SchedulerStepsTest(unittest.TestCase):
    def _make_trainer(self, train_steps=None):
        trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
        trainer.args = SimpleNamespace(
            learning_rate=1e-4,
            beta1=0.9,
            beta2=0.95,
            beta3=0.98,
            epsilon=1e-8,
            weight_decay=1e-4,
            optimizer="adamw",
            gradient_accumulation_steps=2,
            train_steps=train_steps,
            train_epochs=3,
            lr_scheduler="linear",
            lr_warmup_steps=2,
            lr_num_cycles=1,
            lr_power=1.0,
        )
        trainer.state = SimpleNamespace(
            num_trainable_parameters=0,
            overwrote_max_train_steps=False,
            num_update_steps_per_epoch=0,
        )
        trainer.components = SimpleNamespace(transformer=torch.nn.Linear(1, 1))
        trainer.accelerator = _FakeAccelerator(num_processes=4)
        trainer.data_loader = _SizedLoader(length=12)
        return trainer

    def _prepare(self, trainer):
        scheduler_kwargs = {}

        def capture_scheduler(name, optimizer, **kwargs):
            scheduler_kwargs.update(kwargs)
            return object()

        logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
        with (
            patch.object(trainer_module, "get_scheduler", capture_scheduler),
            patch.object(trainer_module, "logger", logger),
        ):
            trainer.prepare_optimizer()
            trainer.prepare_for_training()
        return scheduler_kwargs

    def test_epoch_schedule_uses_prepared_loader_length(self):
        trainer = self._make_trainer()
        scheduler_kwargs = self._prepare(trainer)

        self.assertEqual(trainer.accelerator.prepare_data_loader_calls, 1)
        self.assertEqual(len(trainer.data_loader), 3)
        self.assertEqual(trainer.state.num_update_steps_per_epoch, 2)
        self.assertEqual(trainer.args.train_steps, 6)
        self.assertEqual(scheduler_kwargs["num_training_steps"], 24)
        self.assertEqual(scheduler_kwargs["num_warmup_steps"], 8)

    def test_explicit_train_steps_keep_process_scaling(self):
        trainer = self._make_trainer(train_steps=10)
        scheduler_kwargs = self._prepare(trainer)

        self.assertEqual(trainer.args.train_steps, 10)
        self.assertEqual(scheduler_kwargs["num_training_steps"], 40)
        self.assertEqual(scheduler_kwargs["num_warmup_steps"], 8)


if __name__ == "__main__":
    unittest.main()
