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

"""
File containing the model function and parameterizations for the
UNet Semantic Segmentation model.
"""
import argparse
import os
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from modelzoo.common.tf.estimator.cs_estimator import CerebrasEstimator
from modelzoo.common.tf.estimator.run_config import CSRunConfig
from modelzoo.common.tf.run_utils import (
    check_env,
    get_csconfig,
    get_csrunconfig_dict,
    is_cs,
    save_params,
    update_params_from_args,
)
from modelzoo.vision.tf.unet.data import eval_input_fn, train_input_fn
from modelzoo.vision.tf.unet.model import model_fn
from modelzoo.vision.tf.unet.utils import get_params


def create_arg_parser():
    """
    Create parser for command line args.
    :returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        help="Model directory where checkpoints will be written. "
        + "If directory exists, weights are loaded from the checkpoint file.",
    )
    parser.add_argument(
        "--cs_ip",
        default=None,
        help="IP address of the Cerebras System, defaults to None. Ignored on GPU.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of steps to run mode train."
            + " Runs repeatedly for the specified number."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Number of total steps to run mode train or for defining training"
            + " configuration for train_and_eval. Runs incrementally till"
            + " the specified number."
        ),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help=(
            "Number of total steps to run mode eval, eval_all or for defining"
            + " eval configuration for train_and_eval. Runs once for"
            + " the specified number."
        ),
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["train", "eval", "eval_all", "train_and_eval"],
        help=(
            "Can train, eval, eval_all, or train_and_eval."
            + "  Train and eval will compile and train if on the Cerebras System,"
            + "  and just run locally (CPU/GPU) if not on the Cerebras System."
            + "  train_and_eval will run locally."
            + "  Eval_all will run eval locally for all available checkpoints."
        ),
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Compile model up to kernel matching.",
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        help="Compile model completely, generating compiled executables.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force model to run on a specific device (e.g., --device /gpu:0)",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint to initialize weights from.",
    )

    return parser


def validate_params(params):
    # check validate_only/compile_only
    runconfig_params = params["runconfig"]
    assert not (
        runconfig_params["validate_only"] and runconfig_params["compile_only"]
    ), "Please only use one of validate_only and compile_only."

    # ensure runconfig is compatible with the Cerebras System
    if (
        is_cs(runconfig_params)
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        assert (
            runconfig_params["mode"] == "train"
        ), "For UNET model, only training is supported on the Cerebras System."


def run(
    args, params, model_fn, train_input_fn=None, eval_input_fn=None,
):
    """
    Set up estimator and run based on mode

    :params dict params: dict to handle all parameters
    :params tf.estimator.EstimatorSpec model_fn: Model function to run with
    :params tf.data.Dataset train_input_fn: Dataset to train with
    :params tf.data.Dataset eval_input_fn: Dataset to validate against
    """
    # update runtime params
    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)
    validate_params(params)

    # save params for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])

    tf.random.set_seed(runconfig_params["tf_random_seed"])

    # get cs- specific configurations
    cs_config = get_csconfig(params.get("csconfig", dict()))

    # get runtime configurations
    use_cs = is_cs(runconfig_params)
    params["use_cs"] = use_cs
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)

    # prep cs1 run environment, run config and estimator
    check_env(runconfig_params)
    est_config = CSRunConfig(
        cs_ip=runconfig_params["cs_ip"],
        cs_config=cs_config,
        **csrunconfig_dict,
    )

    est = CerebrasEstimator(
        model_fn=model_fn,
        model_dir=runconfig_params["model_dir"],
        config=est_config,
        params=params,
        warm_start_from=runconfig_params["checkpoint_path"],
    )

    # execute based on mode
    if runconfig_params["validate_only"] or runconfig_params["compile_only"]:
        if runconfig_params["mode"] == "train":
            input_fn = train_input_fn
            mode = tf.estimator.ModeKeys.TRAIN
        else:
            input_fn = eval_input_fn
            mode = tf.estimator.ModeKeys.EVAL
        est.compile(
            input_fn, validate_only=runconfig_params["validate_only"], mode=mode
        )
    elif runconfig_params["mode"] == "train":
        est.train(
            input_fn=train_input_fn,
            steps=runconfig_params["steps"],
            max_steps=runconfig_params["max_steps"],
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "eval":
        est.evaluate(
            input_fn=eval_input_fn,
            checkpoint_path=runconfig_params["checkpoint_path"],
            steps=runconfig_params["eval_steps"],
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "eval_all":
        ckpt_list = tf.train.get_checkpoint_state(
            runconfig_params["model_dir"]
        ).all_model_checkpoint_paths
        for ckpt in ckpt_list:
            est.evaluate(
                eval_input_fn,
                checkpoint_path=ckpt,
                steps=runconfig_params["eval_steps"],
                use_cs=use_cs,
            )
    elif runconfig_params["mode"] == "train_and_eval":
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=runconfig_params["max_steps"]
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=runconfig_params["eval_steps"],
            throttle_secs=runconfig_params["throttle_secs"],
        )
        tf.estimator.train_and_evaluate(est, train_spec, eval_spec)


def main():
    """
    Main function
    """
    
    print("Python version")
    print (sys.version)
    print("tensorflow version")
    print(tf. __version__)
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params)
    run(
        args=args,
        params=params,
        model_fn=model_fn,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
    )


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
