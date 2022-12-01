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
Utils for params parsing for UNet model
"""
import os

import yaml

from modelzoo.common.tf.run_utils import is_cs
try:
    from cerebras.pb.common.tri_state_pb2 import TS_DISABLED, TS_ENABLED
    from cerebras.pb.stack.autogen_pb2 import AP_ENABLED
    from cerebras.pb.stack.full_pb2 import FullConfig
except ImportError:
    pass  # non-cbcore run

# Visualization color codes
color_codes = [
    (0, 0, 0),
    (220, 20, 60),
    (0, 0, 255),
    (244, 35, 232),
    (255, 255, 0),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (0, 0, 142),
    (128, 64, 128),
    (250, 170, 30),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]


_curdir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMS_FILE = os.path.join(_curdir, "configs/params_dagm.yaml")


def get_params(params_file=DEFAULT_PARAMS_FILE):
    """ Returns a dict """
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    set_defaults(params)
    return params


def set_defaults(params):
    params["model"]["nonlinearity"] = params["model"].get(
        "nonlinearity", "ReLU"
    )
    params["model"]["skip_connect"] = params["model"].get("skip_connect", True)
    params["model"]["enable_bias"] = params["model"].get("enable_bias", True)
    params["model"]["data_format"] = params["model"].get(
        "data_format", "channels_first"
    )
    params["model"]["downscale_method"] = params["model"].get(
        "downscale_method", "max_pool"
    )
    params["model"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )

    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", 1.0
    )
    params["optimizer"]["max_gradient_norm"] = params["optimizer"].get(
        "max_gradient_norm", None
    )
    params["optimizer"]["grad_accum_steps"] = params["optimizer"].get(
        "grad_accum_steps", 1
    )
    params["optimizer"]["log_summaries"] = params["optimizer"].get(
        "log_summaries", False
    )

    params["runconfig"]["model_dir"] = params["runconfig"].get(
        "model_dir",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_dir"),
    )
    params["runconfig"]["save_summary_steps"] = params["runconfig"].get(
        "save_summary_steps", 100
    )
    params["runconfig"]["save_checkpoints_steps"] = params["runconfig"].get(
        "save_checkpoints_steps", 1000
    )
    params["runconfig"]["keep_checkpoint_max"] = params["runconfig"].get(
        "keep_checkpoint_max", 5
    )
    params["runconfig"]["tf_random_seed"] = params["runconfig"].get(
        "tf_random_seed", None
    )
    params["runconfig"]["enable_distributed"] = params["runconfig"].get(
        "enable_distributed", False
    )
    params["runconfig"]["multiple_workers"] = params["runconfig"].get(
        "multiple_workers", False
    )
    # handling eval / infer parameters
    params["runconfig"]["throttle_secs"] = params["runconfig"].get(
        "throttle_secs", 0
    )
    params["runconfig"]["predict_steps"] = params["runconfig"].get(
        "predict_steps", 1
    )

    if (
        params["runconfig"]["enable_distributed"]
        and params["optimizer"]["loss_scaling_factor"] == "dynamic"
    ):
        raise ValueError(
            "Cannot use \"dynamic\" loss scaling for distributed training,"
            + " use \"tf_dynamic\" instead."
        )

    params["runconfig"]["log_step_count_steps"] = (
        params["optimizer"]["grad_accum_steps"]
        if params["optimizer"]["grad_accum_steps"] > 1
        else params["runconfig"]["save_summary_steps"]
    )

    params["csconfig"] = params.get("csconfig", {})
    params["csconfig"]["use_cbfloat16"] = params["csconfig"].get(
        "use_cbfloat16", False
    )
    if (
        params["csconfig"]["use_cbfloat16"]
        and not params["model"]["mixed_precision"]
    ):
        raise ValueError(
            "use_cbfloat16=True can only be used in mixed precision"
            " mode. Set mixed_precision to True."
        )



def get_custom_stack_params(params):
    stack_params = {}
    runconfig_params = params["runconfig"]

    if params["model"].get("multireplica"):
        runconfig_params["multireplica"] = True

    if "ir_mode" in runconfig_params:
        stack_params["ir_mode"] = runconfig_params["ir_mode"]
    else:
        stack_params["ir_mode"] = "mlir-cirh"

        model_params = params["model"]
        if (
            "pooler_type" in model_params
            and model_params["pooler_type"] != "first"
        ):
            stack_params["ir_mode"] = "mlir-xla"

    if (
        is_cs(runconfig_params)
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        stack_params["config"] = set_custom_config(FullConfig(), params)

    return stack_params


def set_custom_config(config, params):
    runconfig_params = params["runconfig"]
    config.placement.optimize_buses.deltat_relative_margin = 0.5
    # if params["train_input"]["max_sequence_length"] <= 512:
    #     config.matching.kernel.no_dcache_spill_splits = True
    # if params["train_input"]["max_sequence_length"] > 512:
    #     config.matching.kernel.inc_pwt_estimate = True
    # enable Autogen for extractive summarization model.
    config.matching.autogen_policy = AP_ENABLED
    # Enable multi-replica
    if runconfig_params.get("multireplica"):
        config.target_num_replicas = -1
        config.placement.pathfinder_inter_replica.fix_existing = TS_DISABLED
        config.placement.pathfinder_inter_replica.allow_infeasible_initial_state = (
            TS_ENABLED
        )
        config.matching.match_lair.disabled_converters.append(
            "AttentionCIRHConverter"
        )

    if params["model"].get("use_vsl", False):
        config.matching.kernel.use_legacy_vsl = True

    if runconfig_params.get("mode", "train") == "eval":
        config.matching.add_pack_and_unpack.max_egress_per_pack = 1
        config.placement.prep_recolor_kernels.wrap_pack_kernel = True
        config.matching.match_lair.disabled_converters.append(
            "UnsortedGatherConverter"
        )

    return config
