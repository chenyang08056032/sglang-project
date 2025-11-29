import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_235B_MODEL_PATH,
    QWEN3_235B_OTHER_ARGS,
    QWEN3_235B_ENVS,
)


class TestQwen3_235B(TestSingleMixUtils):
    model = QWEN3_235B_MODEL_PATH
    dataset = (
        "/data/ascend-ci-share-pkking-sglang/modelscope/hub/datasets/Qwen3-235B-A22B-W8A8/GSM8K-in2048-bs5000/test.jsonl")
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    max_out_len = 2048
    batch_size = 78
    num_prompts = int(batch_size) * 4
    tpot = 48.6432
    output_token_throughput = 452.875

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
