# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import HfArgumentParser

from llamafactory.hparams import ModelArguments


def test_v1_kernel_arguments_can_be_parsed():
    parser = HfArgumentParser(ModelArguments)
    (model_args,) = parser.parse_dict({"model_name_or_path": "dummy", "v1_kernel_ids": "te_grouped_gemm"})

    assert model_args.use_v1_kernels is False
    assert model_args.v1_kernel_ids == "te_grouped_gemm"

    (legacy_model_args,) = parser.parse_dict({"model_name_or_path": "dummy", "use_v1_kernels": True})

    assert legacy_model_args.use_v1_kernels is True
    assert legacy_model_args.v1_kernel_ids is None
