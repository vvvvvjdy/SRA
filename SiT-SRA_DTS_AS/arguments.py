import argparse


def _parse_mask_ratio(values):
    if not isinstance(values, (list, tuple)):
        values = [values]

    ratios = []
    for value in values:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [part.strip() for part in text.split(",")] if "," in text else [text]
        for part in parts:
            part = part.strip().strip("\"'")
            if part:
                ratios.append(float(part))

    if not ratios:
        raise argparse.ArgumentTypeError("mask_ratio must contain at least one value")

    return ratios


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--sample-steps", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=801)
    parser.add_argument("--checkpoint-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-epochs", type=int, default=200)
    parser.add_argument("--max-train-steps", type=int, default=4100000)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--attention-separation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="separate attention across token groups when mixed-token training is active",
    )

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    # precision
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=16)

    # loss
    parser.add_argument("--loss-type", type=str, default="cos", choices=["sml1", "l2", "l1", "cos"])
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="use class-free guidance if > 0")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])  # currently we only support v-prediction
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--block-out-s", type=int, default=4)
    parser.add_argument("--block-out-t", type=int, default=8)
    parser.add_argument(
        "--mask-ratio",
        nargs="+",
        default=["0.5"],
        help="mask ratios, e.g. `0.5`, `0.75 0.25`, or `[0.5,0.3,0.2]`",
    )
    parser.add_argument("--use-alignment-loss", action=argparse.BooleanOptionalAction, default=False, help="whether to use the alignment loss")
    parser.add_argument("--align-weight", type=float, default=0.5, help="the weight of the alignment loss")
    parser.add_argument("--teacher-t", type=str, default="self_flow", choices=["self_flow", "same", "sra"])
    parser.add_argument("--teacher-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dual-time-scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="sample independent timesteps for token groups; disabled uses one shared timestep",
    )
    parser.add_argument(
        "--full-sample-prob",
        type=float,
        default=0.0,
        help="probability that a two-group mixed sample is replaced by a full-image sample",
    )

    args = parser.parse_args()
    args.mask_ratio = _parse_mask_ratio(args.mask_ratio)
    if args.full_sample_prob < 0.0 or args.full_sample_prob > 1.0:
        parser.error("--full-sample-prob must be in [0, 1]")

    return args
