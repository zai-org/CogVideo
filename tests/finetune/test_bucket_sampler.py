import importlib.util
from pathlib import Path


def _load_bucket_sampler_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "finetune" / "datasets" / "bucket_sampler.py"
    spec = importlib.util.spec_from_file_location("bucket_sampler_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Dataset:
    def __init__(self, rows):
        self.rows = rows
        self.video_resolution_buckets = [
            (
                row["video_metadata"]["num_frames"],
                row["video_metadata"]["height"],
                row["video_metadata"]["width"],
            )
            for row in rows
        ]
        self.video_resolution_buckets = list(dict.fromkeys(self.video_resolution_buckets))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _row(index, bucket=(1, 32, 32)):
    return {
        "id": index,
        "video_metadata": {
            "num_frames": bucket[0],
            "height": bucket[1],
            "width": bucket[2],
        },
    }


def _flatten_ids(batches):
    return [row["id"] for batch in batches for row in batch]


def test_shuffle_permutates_dataset_before_bucketing(monkeypatch):
    module = _load_bucket_sampler_module()
    dataset = _Dataset([_row(i, (1, 32 + i % 2, 32)) for i in range(8)])
    sampler = module.BucketSampler(dataset, batch_size=2, shuffle=True)
    shuffle_calls = []

    def reverse(values):
        shuffle_calls.append(len(values))
        values.reverse()

    monkeypatch.setattr(module.random, "shuffle", reverse)
    batches = list(sampler)

    assert shuffle_calls[0] == len(dataset)
    assert sorted(_flatten_ids(batches)) == list(range(len(dataset)))
    assert _flatten_ids(batches) != list(range(len(dataset)))
    for batch in batches:
        metadata = {
            (
                row["video_metadata"]["num_frames"],
                row["video_metadata"]["height"],
                row["video_metadata"]["width"],
            )
            for row in batch
        }
        assert len(metadata) == 1


def test_drop_last_does_not_carry_partial_bucket_between_epochs():
    module = _load_bucket_sampler_module()
    dataset = _Dataset([_row(i) for i in range(3)])
    sampler = module.BucketSampler(dataset, batch_size=2, shuffle=False, drop_last=True)

    first_epoch = list(sampler)
    second_epoch = list(sampler)

    assert _flatten_ids(first_epoch) == [0, 1]
    assert _flatten_ids(second_epoch) == [0, 1]
