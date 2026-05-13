import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Thêm các tham số riêng của EDR-REDNet vào argument parser.

    Parameters
    ----------
    loss_alpha : float
        Trọng số của SobelEdgeLoss trong CombinedLoss (default = 0.1).
        L_total = L_Charbonnier + alpha * L_Sobel
    loss_beta : float
        Trọng số của PerceptualLoss (VGG) — chỉ dùng trong ablation (default = 0.0).
    loss_gamma : float
        Trọng số của HU Loss — BẬT sau khi định nghĩa inverse norm (default = 0.0).
    num_edge_blocks : int
        Số lượng EdgeDilatedResidualBlock ở bottleneck (default = 2).
    """
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.1,
        help="Weight for SobelEdgeLoss in CombinedLoss (default = 0.1)",
    )
    parser.add_argument(
        "--loss_beta",
        type=float,
        default=0.0,
        help="Weight for Perceptual (VGG) Loss — ablation only (default = 0.0 = OFF)",
    )
    parser.add_argument(
        "--loss_gamma",
        type=float,
        default=0.0,
        help="Weight for HU Loss — enable after inverse norm is defined (default = 0.0 = OFF)",
    )
    parser.add_argument(
        "--num_edge_blocks",
        type=int,
        default=2,
        help="Number of EdgeDilatedResidualBlocks at bottleneck (default = 2)",
    )
    parser.add_argument(
        "--use_sobel_input",
        action="store_true",
        default=True,
        help="Whether to use FixedSobelLayer features at bottleneck (default = True)",
    )
    parser.add_argument(
        "--no_sobel_input",
        action="store_false",
        dest="use_sobel_input",
        help="Disable FixedSobelLayer features for ablation study",
    )
    return parser
