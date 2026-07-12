# PS2D v2 estimator 迁移与无噪声验证结果

状态：identity/pure-EoR 链条通过；当前 compiled nuisance projector 未通过相对 raw
foreground avoidance 的晋级门，因此停止在 8 频无噪声阶段，不运行 16 频、噪声、split
cross-power 或 full-Fisher 扩展。

## 1. 冻结契约与三套布局

本轮所有 signal、probe、transfer 和 evaluator 共用：

- analysis contract：
  `ce60b514464478c8d5543850805cc5f417f2bcae43f192e42adec06381bd8e64`；
- estimator contract：
  `7771237e46a5a866ebe7fb988a7f5b4963a76c80d000b1d1248213f39aa29e58`；
- 32 个 mode-first science bands，共 31,512 个 FFT modes；
- control 为原生 $k_\parallel$ index 0，guard 为 index 1；两者不与 science selector
  混用。

估计器显式维护三套用途不同的布局：

1. `full` 布局用于 control/guard 与完整诊断；
2. `science` 布局只含逐 mode 通过 floor、wedge、buffer 和 UV support 的 32 个 bands；
3. `calibration source` 布局由 32 个 science source bands 加 146 个互斥补集 bands
   构成，覆盖全部 524,288 个三维 FFT modes。

补集按 control、guard、science-$k_\parallel$ 中的窗口外模式和 radial Nyquist 分类。
因此 probe window 的每一行都能归一到完整输入空间，不会把 taper 从窗口外卷入的功率
误记为丢失。32 个 science bands 与 146 个补集 bands 互斥且完备；partial science band
也不会与补集混在同一 source column 中。

## 2. Identity 与 pure-EoR 闭环

在真实 8 频 dirty-EoR cube 上，先使用 identity projector 校验 estimator 代数：

| 指标 | 结果 |
| --- | ---: |
| identity transfer 最大单位阵误差 | $3.349\times10^{-15}$ |
| pure-EoR row-normalized 相对 L2 | $1.489\times10^{-15}$ |
| pure-EoR 精确 mode-sum power ratio | 1.000000 |
| NumPy/Torch bandpower 最大相对误差 | $3.581\times10^{-15}$ |
| 64 probes/source 与解析 window 的 row-max 差异中位数 | 0.002326 |
| 上述差异最大值 | 0.012534 |

所有预登记 identity gates 均通过。这里的解析 source response 直接计算“输入 Fourier
source mask经过 inverse FFT、global demean、Hann taper、再 FFT”产生的卷积，不使用
EoR realization 或其功率谱；Monte Carlo probes 只是对该解析结果做独立回归。

## 3. Hann taper 下的 source window

即使 projector 为 identity，三轴 Hann taper 也使一个输出 science band 接收窗口外输入：

| source-window 统计量 | 中位数 | 最差值 |
| --- | ---: | ---: |
| self response | 0.510539 | 最小 0.303167 |
| 全部 science-source fraction | 0.789496 | 最小 0.530221 |
| 全部补集 fraction | 0.210504 | 最大 0.469779 |
| control fraction | 0.000140 | 最大 0.002171 |
| guard fraction | 0.002171 | 最大 0.206164 |
| science-$k_\parallel$ 窗口外 fraction | 近 0 | 最大 0.261446 |
| radial-Nyquist fraction | 0.206023 | 该项为主要中位补集来源 |

按事先固定的 `self >= 0.5`、`far leakage <= 0.25`，纯分析算子的 truth-blind support
为 19/32 bands。这里的 radial-Nyquist 贡献说明“分析时排除该层”不等于“taper 前该层对
输出没有响应”；完整 source basis 正是为显式记录这种混合而设置。

## 4. Compiled nuisance 无噪声 screen

随后加载 canonical 8wide、full64、absolute-mask、C0/block8、64 参数 compiled response。
只使用 control modes 拟合 nuisance，guard 与 science modes 不参与系数求解；系数冻结后
才评估 32 个 science outputs。response calibration 与 support gates 不读取 injected EoR。

64 probes/source 的结果为：

- analysis support 19/32，nuisance 后 support 18/32；
- nuisance rank 64/64；transfer rank 32/32，保留子空间条件数 1.0876；
- pure-EoR 在 18-band support 上的相对 L2 为 0.04168，power ratio 为 0.98481；
- current reference 的 foreground leakage 仅为 EoR power 的约 0.00181；
- 但 current corrected L2 为 0.03558，明显差于 raw avoidance 的 0.00877；
- C0-$10^{-6}$ 与 oracle reference 也分别从 raw 0.00700/0.00836 恶化到
  0.03792/0.03585。

因此失败不是 transfer rank、条件数或残余 foreground leakage，而是 nuisance projection
造成的 EoR loss。唯一失败的晋级门是 `current_not_worse_than_raw`。

## 5. Probe 稳定性与冻结 support

32 和 64 probes/source 各得到 18 个 supported bands，但有 2 个 band 的对称差；两次
truth-blind support 的稳定交集为 17 bands、13,192 个 FFT modes。projected source window
的 row-max 差异中位数/最大值为 0.00469/0.01598，说明 window 数值本身稳定，但阈值附近
仍有两个分类变化。

在冻结的 17-band 交集上：

| 输入/reference | raw L2 / power ratio | corrected L2 / power ratio |
| --- | ---: | ---: |
| pure EoR | -- | 0.04271 / 0.97630 |
| current | 0.00896 / 1.00851 | 0.03659 / 0.98202 |
| C0-$10^{-6}$ | 0.00715 / 1.00705 | 0.03890 / 0.97868 |
| oracle | 0.00853 / 1.00818 | 0.03686 / 0.98236 |

稳定 support 在冻结后统计得到：包含 mode-first science-window dirty-EoR power 的
30.12%，约为完整诊断平面 dirty-EoR power 的 3.03%。这些比例只用于影响审计，不参与
support 生成。最关键的结论仍是 corrected L2 比 raw 大约高四倍。

## 6. 弱 ridge 检查与停止决定

为排除 hard projection 过强这一解释，预先检查了 ridge $10^{-6}$、$10^{-5}$ 和
$10^{-4}$。current corrected L2 依次为 0.04869、0.03655、0.01626，仍均差于共同的
raw 0.00858；对应 control residual power ratio 从 0.1518 增至 0.2698。增大 ridge
只是逐渐回到“不修正”，既没有扩大 support，也没有超过 raw baseline。

按预登记停止门，本候选不晋级：

- 不运行 16wide；
- 不加入 thermal noise 或 split cross-power；
- 不为已失败的 projector 构造昂贵的 full Fisher；
- 不根据 EoR truth 扫描更多 ridge 或调整 support threshold。

这否定的是当前 control-only compiled nuisance projection，不是否定 PS2D v2 estimator
或所有 full-covariance quadratic estimator。后续若提出真正不同的
$B S_b B^\dagger$ full-covariance 方法，必须先给出新的 truth-free 可识别性理由，并把
raw foreground avoidance 保留为必须超过的基线。

## 7. 可复现入口

- identity/pure-EoR evaluator：
  `3dnet/ops_scripts/validate_ps2d_v2_estimator_identity.py`；
- compiled nuisance evaluator：
  `3dnet/ops_scripts/evaluate_compiled_nuisance_ps2d_v2_noiseless.py`；
- probe 稳定性汇总：
  `3dnet/ops_scripts/summarize_ps2d_v2_probe_stability.py`；
- identity 结果：
  `runs/ps2d_v2_estimator_validation_20260712/8wide_identity_probe64_analytic_v2/`；
- compiled/probe/ridge 结果：
  `runs/compiled_nuisance_ps2d_v2_20260712/`。
