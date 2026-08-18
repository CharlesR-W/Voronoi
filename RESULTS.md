# Measured results

Last verified: 2026-08-17 (US/Pacific)

## Experiment 1 collection pilot

The canonical run `exp1-cpu-seed0-20260817-v4` completed the requested first collection
pass on one synthetic task and CIFAR-10. It used checkpoints 0, 1, 5, 20, and 100, a
fixed seed-0 probe bank, and matched $128\times16\times16$ transition inputs:

- ResNet: `stage2.block1 -> stage2.block2`, a residual transition;
- VGG-19+BN: `stage2.conv1 -> stage2.conv2`, a non-residual comparison; and
- synthetic: block 1 of a width-32, four-block, normalization-free residual MLP on a
  three-class 2D Gaussian mixture.

For each real model/checkpoint shard the collector saved 256 covariance-fit site
vectors, four real and four covariance-Gaussian centers, eight paired directions per
center, 37 path points, $17\times17$ local planes, three real anchors on a
$21\times21$ affine grid, eight Hutchinson probes, and the complete frozen activation
contexts needed to replay the interventions. Exact float32 path, local-grid, and
anchor-grid intervention vectors are stored rather than reconstructed later.

## Animations

Each GIF has a fixed 1440x900 layout and global scales, five synchronized checkpoint
frames plus a conclusion hold, and decoded durations of 3000, 1000, 1000, 1000, 1000,
and 6000 ms. A static final-frame PNG and machine-readable rendering metadata accompany
each GIF. Synthetic and ResNet Jacobian panels display the two-direction,
plane-restricted residual update $D(T-I)$; VGG panels display the corresponding raw
transition $DT$. Both raw and residual-adjusted fields remain stored where applicable.

| View | GIF | Static fallback |
|---|---|---|
| Synthetic real versus fake | [GIF](docs/assets/experiment1/synthetic_real_fake.gif) | [PNG](docs/assets/experiment1/synthetic_real_fake_final.png) |
| CIFAR ResNet real versus fake | [GIF](docs/assets/experiment1/cifar_resnet_real_fake.gif) | [PNG](docs/assets/experiment1/cifar_resnet_real_fake_final.png) |
| CIFAR VGG real versus fake | [GIF](docs/assets/experiment1/cifar_vgg_real_fake.gif) | [PNG](docs/assets/experiment1/cifar_vgg_real_fake_final.png) |
| CIFAR three-anchor RGB and Jacobians | [GIF](docs/assets/experiment1/cifar_architecture_cells.gif) | [PNG](docs/assets/experiment1/cifar_architecture_cells_final.png) |

The animation validator reopens all media, checks the exact schedule and canvas,
requires fixed/global scales and coordinates, recomputes the three-channel RGB
normalization, and replays the displayed scalar-field extrema from the raw child
artifacts.

## Sanity-check observations

The synthetic classifier reaches 100% test accuracy at epoch 1 and remains there
through epoch 100. This verifies that the collection pipeline spans an actual learned
trajectory; it does not provide a planted plateau or cell ground truth.

For the response curves, define

$$
q_{1/2} = \frac{\operatorname{median} R(\alpha=0.5)}
                 {\operatorname{median} R(\alpha=1)},
$$

where $R$ is logit $L_2$ displacement and the median is over the four centers and eight
directions. The small pilot mostly produces roughly proportional response rather than
an obvious flat-then-jump curve: synthetic $q_{1/2}$ stays between 0.482 and 0.507;
CIFAR ResNet spans 0.483--0.692; and CIFAR VGG spans 0.432--0.664 across center kinds
and checkpoints. These ranges are descriptive checks from 32 paths per center kind,
not a plateau test or uncertainty analysis.

The endpoint medians below are useful for scale checking, not architecture comparison:

| Epoch | ResNet real | ResNet fake | VGG real | VGG fake |
|---:|---:|---:|---:|---:|
| 0 | 0.03394 | 0.02312 | 0.0000287 | 0.0000215 |
| 1 | 0.42165 | 0.20689 | 0.08074 | 0.04243 |
| 5 | 0.68217 | 0.66849 | 0.07008 | 0.06334 |
| 20 | 1.17287 | 0.85333 | 0.08341 | 0.04926 |
| 100 | 1.93487 | 1.92694 | 0.03484 | 0.01814 |

At epoch 100, the mean displayed local-plane norms for real versus fake centers are
0.16449 versus 0.15465 for synthetic $D(T-I)$, 0.34959 versus 0.29724 for ResNet
$D(T-I)$, and 0.34997 versus 0.32121 for VGG $DT$. The fake/real ratios across epochs
are nonmonotonic for both CIFAR architectures, and the ResNet normalized path response
is essentially equal at epoch 100 despite its plane-field difference. This is another
reason not to interpret the visual contrast as a detected plateau.

The raw same-site ResNet next-transition Jacobian Frobenius estimate is approximately
11.3--11.8, while the stored residual-adjusted $\|J-I\|_F$ medians are 0.987--3.53.
This confirms that the identity skip materially dominates the unadjusted norm and is
why both values are saved. VGG has no $J-I$ quantity; its corresponding unadjusted
estimate falls from roughly 3.8 at initialization to 1.5--1.7 late in training. These
numbers use different learned functions and activation scales and do not identify a
residual-architecture effect.

## Interpretation limits

The requested visualization combined distinct source methods. The precise mapping,
official source links, and local audit-copy hashes are in
[references/activation-plateau-source-audit.md](references/activation-plateau-source-audit.md).
Only the path responses and three-context RGB construction are source analogues. The
checkpoint-time GIFs, plane-restricted derivatives, and full same-site Hutchinson
estimates are new hybrid diagnostics.

The data are single-seed and use legacy checkpoint trajectories. Four centers per kind
are enough to exercise and inspect the machinery, but not to establish a population
effect. ResNet and VGG differ in more than their skip connections. No stable-cell,
Voronoi, fissure, basin-commitment, flatness/plasticity, or causal claim should be
inferred from this run. The all-MNIST integrated-Jacobian-barrier visualization is
planned but has not been collected.

See [artifacts/EXPERIMENT_1_DATA.md](artifacts/EXPERIMENT_1_DATA.md) for exact artifact
IDs, child shards, replay fields, and verification commands.
