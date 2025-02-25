# Copyright 2022 Cerebras Systems.
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

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from modelzoo.common.pytorch.run_utils import run
from modelzoo.transformers.pytorch.bert.fine_tuning.qa.data import (
    eval_input_dataloader,
    train_input_dataloader,
)
from modelzoo.transformers.pytorch.bert.fine_tuning.qa.model import (
    BertForQuestionAnsweringModel,
)
from modelzoo.transformers.pytorch.bert.fine_tuning.qa.utils import set_defaults


def main():
    run(
        BertForQuestionAnsweringModel,
        train_input_dataloader,
        eval_input_dataloader,
        default_params_fn=set_defaults,
    )


if __name__ == '__main__':
    main()
