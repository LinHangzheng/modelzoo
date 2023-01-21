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

#!/usr/bin/env python3

"""
Run script for running on cerebras appliance cluster
"""

import os
import sys

import tensorflow as tf

# Disable eager execution
tf.compat.v1.disable_eager_execution()

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

<<<<<<< HEAD
from cerebras_tensorflow.cs_estimator_app import CerebrasAppEstimator

from cerebras_appliance.cs_run_config import CSRunConfig
from cerebras_appliance.CSConfig import CSConfig
from modelzoo.common.tf.appliance_utils import (
    get_debug_args,
    parse_args_and_params,
)
from modelzoo.common.tf.run_utils import get_csrunconfig_dict
=======
from modelzoo.common.tf.appliance_utils import ExecutionStrategy, run_appliance
>>>>>>> a3bf8f62b2f2e46d0d9ae688911596df52a36168
from modelzoo.transformers.tf.gpt2.data import eval_input_fn, train_input_fn
from modelzoo.transformers.tf.gpt2.model import model_fn
from modelzoo.transformers.tf.gpt2.utils import set_defaults


def main():
<<<<<<< HEAD
    run_dir = os.path.dirname(os.path.abspath(__file__))

    # Parser cmdline arguments and get params
    params = parse_args_and_params(run_dir, set_default_params=set_defaults)
    runconfig_params = params["runconfig"]
    with tempfile.TemporaryDirectory(dir=run_dir) as ini_dir:
        with open(os.path.join(ini_dir, "debug.ini"), "w") as fp:
            fp.write("ws_dense_dmatmul: true")
        debug_args = get_debug_args(ini_dir)

    # log settings
    logging.info(f'Credentials path: {runconfig_params["credentials_path"]}')
    logging.info(f"Debug args: {debug_args}")

    # figure out the input_fn to use
    if runconfig_params["mode"] == "train":
        input_fn = train_input_fn
        mode = tf.estimator.ModeKeys.TRAIN
    elif runconfig_params["mode"] == "eval":
        input_fn = eval_input_fn
        mode = tf.estimator.ModeKeys.EVAL
    else:
        raise ValueError(f'Mode not supported: {runconfig_params["mode"]}')

    # create the run config
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)
    cs_run_config = CSRunConfig(
        cs_config=CSConfig(
            num_csx=runconfig_params["num_csx"],
            max_wgt_servers=runconfig_params["num_wgt_servers"],
            mgmt_address=runconfig_params["mgmt_address"],
            credentials_path=runconfig_params["credentials_path"],
            debug_args=debug_args,
            mount_dirs=[
                runconfig_params["data_mount_path"],
                runconfig_params["modelzoo_mount_path"],
            ],
            python_paths=[runconfig_params["modelzoo_mount_path"]],
        ),
        **csrunconfig_dict,
    )

    # create estimator
    cs_estimator = CerebrasAppEstimator(
=======
    run_appliance(
>>>>>>> a3bf8f62b2f2e46d0d9ae688911596df52a36168
        model_fn,
        train_input_fn,
        eval_input_fn,
        supported_strategies=[
            ExecutionStrategy.weight_streaming,
            ExecutionStrategy.pipeline,
        ],
        default_params_fn=set_defaults,
    )


if __name__ == '__main__':
    main()
