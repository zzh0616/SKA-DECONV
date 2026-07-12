# PS2D v2 estimator 迁移与无噪声验证结果

状态：identity/pure-EoR 链条通过；control-only compiled nuisance projector 仍被拒绝。
随后独立实现的 reduced full-covariance sky-operator QML 已在 8 频无噪声、6 个预登记
target bands 上通过初级门槛，但尚未完成 16 频、噪声、split cross-power 或 uncertainty
扩展，因此不能作为 production power-spectrum 结果。

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

## 8. Reduced full-covariance QML 后续测试

### 8.1 方法与 target 冻结

旧 projector 失败后，没有继续调 ridge 或 support threshold，而是实现了一个不同的
低维完整协方差 screen：

- 从 32 个 mode-first science bands 中预先选取 6 个几何分层 target bands
  `[4, 10, 16, 7, 13, 19]`；
- 对 target groups 各保留最多 32 个 conjugate-unique Fourier representatives，对
  context groups 各保留最多 64 个；real/imag 展开后共有 768 个 real features；
- 对每个 intrinsic-sky source group 生成单位功率 Gaussian probes，先经过相同的
  cached stride4/rank64 PCA forward operator，再测量 reduced features，由此估计完整的
  $B S_b B^\dagger$，而不是只保留对角 PSD；
- signal covariance 使用 36 个自由幅度：6 个 targets、其余 26 个 science bands
  分别建模，以及 control、guard、window 外和 radial-Nyquist 四个非 science groups；
- compiled foreground response 以有限 covariance 进入 weighting matrix。其尺度只在
  control features 上按 Gaussian NLL 选择；最终结果不减去假设的 foreground covariance
  bias，即 `bias_subtraction_scope=none`，避免把未知前景幅度当成已知真值；
- bandpower 使用 full-covariance quadratic/Fisher 解。support 由 Fisher window 和
  generalized signal fractions 冻结，不读取 injected EoR realization。

这里的 36 个 bandpower/covariance amplitudes 都由当前数据估计，并不是用训练集预先给定
的 EoR shape。sky probes 只标定线性 operator 对单位 source covariance 的 response。

### 8.2 Operator closure 与早期失败定位

在真实 8 频 EoR OSM sky 上，把完整 sky 通过 cached PCA operator 的输出与 exact dirty
FITS 比较，得到：

| 闭环诊断 | relative L2 | cosine |
| --- | ---: | ---: |
| dirty cube | $2.28596\times10^{-6}$ | $0.999999999997$ |
| 768 reduced features | $2.63086\times10^{-6}$ | $0.999999999997$ |

因此 rank64 PCA proxy 在这个测试上的 forward closure 足够准确。相反，最初只建模 6 个
target covariances 时，target-only features 与 exact features 的 relative L2 为
`0.69366`，遗漏 context 的 feature norm 是 target contribution 的 `0.93673`。这解释了
为何最早的 full-covariance 结果很差：问题是 taper/operator 混合后遗漏了大幅 signal
context，而不是 forward operator 数值错误。

### 8.3 单视图、无真值选参和 controls

第一视图使用固定 feature seed `20260712`。sample covariance 向 diagonal shrink 的候选
只取预先固定的 `0.25/0.5/0.75/1.0`。选择指标为当前 observed feature vector 的

$$
\frac{1}{2}\left[\log\det C + x^\mathsf{T}C^{-1}x\right],
$$

其 NLL 依次为 `-7018.94/-7022.79/-7017.05/-7009.27`，故选择 `0.5`。选择过程不读取
foreground truth 或 EoR truth。

| 输入/reference | count-weighted L2 | integrated power ratio |
| --- | ---: | ---: |
| pure EoR | 0.25491 | 1.05868 |
| current | 0.27890 | 1.12394 |
| C0-$10^{-6}$ | 0.27910 | 1.12474 |
| oracle | 0.27826 | 1.12307 |

6/6 targets、18,104 个 FFT modes 被保留，held-out probe-ensemble target-row response
最大单位阵误差为 `0.30137`。按预登记聚合门槛，pure/current 均满足 L2 不超过 0.3、
integrated power 在 $1\pm0.3$ 内，controls 也没有显示 reference 敏感性。这里的门槛是按
mode count 加权的整体诊断，并不意味着每个单独 band 的误差都小于 30%。

### 8.4 独立视图与固定多视图平均

为检查 probe/feature realization 稳定性，又以独立 feature seed `20260713` 和独立 sky
probes 重建完整 36-source bank。它自己的 observed-data NLL 在同一固定网格上选择
shrinkage `0.75`，但该视图单独失败：pure/current L2 为 `0.62144/0.63891`，功率比为
`0.96523/0.95864`。这证明单视图 sample covariance 仍有明显 realization variance，不能
只报告通过的 seed。

随后只组合两个视图各自无真值选择出的估计，并使用预先固定的等权 `0.5/0.5`，不再拟合
组合权重：

| 输入/reference | count-weighted L2 | integrated power ratio |
| --- | ---: | ---: |
| pure EoR | 0.25637 | 1.01195 |
| current | 0.27458 | 1.04129 |

共同 support 为 6/6，ensemble target-row response 最大误差为 `0.27484`，所有聚合晋级门
通过。这个结果说明独立 probe views 的平均可以降低 covariance Monte Carlo 方差，但第二
视图单独失败仍是需要通过增加 probes/频率和独立复制解决的风险。

### 8.5 当前结论与停止边界

本轮只建立了 8wide、无噪声、synthetic auto-power、6-band feasibility。它没有证明：

- map-level EoR recovery；
- 完整 EoR window 或逐 band 的 30% 精度；
- thermal-noise bias 已被消除；
- split cross-power、误差条和 coverage 已校准；
- 对真实 foreground/calibration mismatch 稳健。

最终 cache 审计只找到 36 GiB 的 stride4/rank64 tmpfs bank，实际包含
`117.9, 118.3, ..., 120.7 MHz` 八个频率；旧 16wide products 是不同 selected-mask
契约，不能直接当作本方法的完整 16wide bank。因此没有盲目启动扩频。下一步应先冻结同一
feature/source/operator contract，补齐 16wide cache 并做独立 replication；通过后再进入
noise splits、cross-power、Fisher/Monte-Carlo uncertainty 和 coverage tests。

新增复现入口：

- `3dnet/ps2d_v2_full_covariance.py`；
- `3dnet/ops_scripts/build_ps2d_v2_sky_band_operator_features.py`；
- `3dnet/ops_scripts/evaluate_ps2d_v2_full_covariance_reduced.py`；
- `3dnet/ops_scripts/diagnose_ps2d_v2_sky_operator_closure.py`；
- `3dnet/ops_scripts/select_ps2d_v2_shrinkage_by_data_nll.py`；
- `3dnet/ops_scripts/combine_ps2d_v2_multiview_estimates.py`；
- `runs/ps2d_v2_full_covariance_20260712/`。
