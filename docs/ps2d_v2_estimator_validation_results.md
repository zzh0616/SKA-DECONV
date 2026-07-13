# PS2D v2 estimator 迁移与无噪声验证结果

状态：identity/pure-EoR 链条通过；control-only compiled nuisance projector 仍被拒绝。
reduced full-covariance sky-operator QML 的早期聚合 PASS 已被严格受控测试撤销：随机
feature/probe covariance 未收敛，16→32 probes 也没有改善。flat-prior template T0/T1
mean profile 成功消除了 reference 幅度/倾斜依赖，但逐 band 与跨视图稳定性仍失败。因此
它只保留为诊断。后续无 fixed-template sky-smooth flat-projection screen 同样失败：T0
欠拟合，T0/T1 在前景远未压净时已经损失 11.5% 目标 EoR 总功率。因此不运行 64/128
probes、16 频、噪声、split cross-power 或 uncertainty 推广。后续无固定形态的有限
sky-smooth covariance MAP screen 也已完成：guard-only selector 退化到近似 flat 的低-ridge
极限，前景残留仍高约六个数量级。因此下一候选只能改为 likelihood-level covariance
marginalization 或保守 avoidance。

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

## 9. 2026-07-13 严格受控稳定性复核

### 9.1 为什么撤销上一节的聚合 PASS

第 8 节的 L2 和总功率指标会被高功率 bands 主导，而且两个独立视图存在明显误差抵消。
因此冻结了一个更严格的后续 screen：

- aggregate L2 不超过 0.2，总功率比在 $1\pm0.1$；
- 6 个 target 必须全部为正且每个相对偏差不超过 30%；
- held-out target-row response 最大误差不超过 0.3；
- view Monte Carlo 标准误不超过对应 truth 的 20%；
- 同 probe/异 feature、同 feature/异 probe 和 probe 翻倍的 truth-free weighted change
  均不超过 0.15；
- 最终 science gate 另行记录逐 band 10% 目标，不与 screen gate 混用。

原 view12+13 在新口径下 pure/current 最大逐 band 偏差为 `0.62648/0.67448`，最大 MC
标准误为 `1.01935`，跨视图变化为 `0.72448/0.74768`，所以旧 PASS 不能保留。

### 9.2 受控 feature/probe 方差分解

在 119 的四张 A800 上构建了两套完整 36-source banks，均使用 probe seed `20260720`，
feature seed 分别为 `20260712` 和 `20260713`。每个 source 含 16 calibration 与 16
held-out probes。再把 feature13/probe20 与旧 feature13/probe19 合并为 32-probe bank。
每个 view 仍只用 observed-data NLL 从固定 `0.25/0.5/0.75/1.0` 网格选 shrinkage。

covariance-only 三视图等权结果为：

| 指标 | pure EoR | current |
| --- | ---: | ---: |
| aggregate L2 | 0.28376 | 0.30676 |
| integrated power ratio | 0.92988 | 0.98808 |
| 最大逐 band 偏差 | 0.66355 | 0.80365 |

response 最大误差为 `0.33926`，最大 MC 标准误比例为 `0.49635`，严格 screen 失败。
受控比较进一步定位了来源：

| 只改变的因素 | pure change | current change | response change |
| --- | ---: | ---: | ---: |
| feature seed 12→13，同 probe20 | 0.40976 | 0.41185 | 0.23046 |
| probe seed 19→20，同 feature13 | 0.81734 | 0.79473 | 0.18493 |
| feature13 probes 16→32 | 0.66939 | 0.63361 | 0.06572 |

固定 shrinkage `0.5` 或 `0.75` 后，16→32 的 pure/current change 仍约为
`0.74--0.79/0.69--0.75`。所以失败不是 NLL selector 跨格跳变。32-probe view 本身的
pure/current L2 为 `0.54953/0.58241`，也没有显示收敛趋势。按预登记门槛停止 64/128
probes；不能继续生成很多 seeds 后按 truth 挑选较好的 realization。

### 9.3 Foreground reference controls

从 current deployable dirty template 构造三种不读取 foreground/EoR truth 的 stress：

- 零 reference；
- template 整体乘 0.9；
- 频率方向乘 $1+0.1\,\mathrm{linspace}(-1,1,N_\nu)$。

原 covariance-only QML 对三者的 L2 分别达到约 `1.50e6`、`1.50e4` 和 `4.67e4`；原来
的 aggregate PASS 强烈依赖非常接近 current 的 reference。把 64 个 compiled correction
columns 作为 flat-prior mean profile 仍不能吸收这些误差，而且 NLL-selected current L2
退化到 `0.40088`。

随后加入从 current template 本身构造的全局 Chebyshev mean columns。profile compiled
columns 加 template T0/T1/T2 可以让四种 reference 得到相同结果，但 pure/current L2
仍为 `0.36387/0.36279`。更小的 template-only T0/T1 profile 更有效：只解析 profile
template 幅度和线性 tilt，同时保留 compiled correction covariance 作为 weighting。
它不使用 simulation truth，也不给系数强 prior。

在旧 view12 上，T0/T1 profile 的四种 reference 数值完全一致：

| 指标 | pure EoR | current/zero/amp90/tilt10 |
| --- | ---: | ---: |
| aggregate L2 | 0.24668 | 0.27531 |
| integrated power ratio | 1.05236 | 1.13779 |

这证明 reference 幅度和线性谱倾斜的不变性已工程闭环。但是在受控三视图上，它只把
pure/current aggregate L2 改善为 `0.19950/0.24553`；最大逐 band 偏差仍为
`0.48835/0.77090`，最大 MC SE `0.58911`。同 probe 异 feature、同 feature 异 probe、
16→32 comparisons 仍全部失败。

### 9.4 最终决定

flat-prior fixed-template T0/T1 profile 只保留为 reference-robustness 诊断，不再作为未来
estimator 的默认 mean component，因为固定 foreground morphology 的假设过强。当前 768
random-feature/sample-covariance QML 同样被拒绝；其失败不是 forward operator closure、
foreground reference 幅度或 NLL selector，而是 covariance 与 reduced-feature 近似本身
没有收敛。

因此不继续：

- 64/128 probe 扫描；
- 增加 random Fourier representatives；
- 16wide cache/replication；
- thermal-noise split、cross-QML 或 uncertainty coverage。

下一方法必须使用 analytic/deterministic covariance，或真正 matrix-free 的 full-plane
$B S_b B^\dagger$ actions，并先通过同一受控 8wide 门。新增入口：

- `3dnet/ps2d_v2_stability.py`；
- `3dnet/ops_scripts/evaluate_ps2d_v2_view_stability.py`；
- `3dnet/ops_scripts/build_foreground_reference_stress_controls.py`；
- `3dnet/ops_scripts/evaluate_ps2d_v2_controlled_views_8wide.sh`；
- `runs/ps2d_v2_controlled_stability_20260713/`；
- `runs/ps2d_v2_foreground_controls_20260713/`。

## 10. 无模板 sky-smooth nuisance screen

### 10.1 模型与闭环

为了避免 fixed-template 的空间形态先验，新候选直接使用
$f(\nu,x,y)=\sum_m q_m(\nu)A_m(x,y)$。每个正交 Chebyshev 谱基列对应一张自由
`512×512` sky map，不使用 foreground reference。T0/T1 名称在这里仅指谱方向的常数和
线性列。foreground 与 EoR 先相加形成唯一 observed cube，求解过程不读取分量标签。

cached stride4/rank64 operator 直接放入 matrix-free LSQR，adjoint 用同一 forward graph
的 autograd action。T0 和 T0/T1 dot-product closure 分别为 `0` 与 `4.24e-16`。其系数/
dirty-data 维数比为 `0.5/1.0`，所以 T0/T1 flat subspace 不存在受保护的维数 null space。

### 10.2 无噪声结果

| 谱空间 | iterations | data residual | observed target L2 / power ratio | pure-EoR target L2 / power ratio |
| --- | ---: | ---: | ---: | ---: |
| T0 | 24 | 0.08531 | `7.692e5 / 1.179e6` | `0.03153 / 0.97143` |
| T0/T1 | 24 | 0.05201 | `2.136e5 / 4.584e5` | `0.12452 / 0.88515` |
| T0 | 96 | 0.03787 | `1.699e5 / 3.144e5` | 未运行 |
| T0/T1 | 96 | 0.01971 | `9.002e4 / 1.276e5` | 未运行 |

T0/T1 在前景仍远未压净时已经移除 `11.5%` 的 frozen-target EoR 总功率，最差单 band
损失 `12.47%`。96 次拟合继续降低 data residual，却只把 observed target L2 降到
`9.002e4`，不能靠增加迭代补足四至五个数量级。8-probe Hutchinson Jacobi 估计约
`47.6%` 对角为非正或非有限，结果没有改善，故保持关闭。该 pipeline 是 deterministic
full-cube solve，不存在 random feature/probe view 选择。

### 10.3 决定

停止 identity-weighted flat hard projection，不晋级更多频率、noise splits 或 cross-power。
这不否定 template-free sky-smooth nuisance 本身；下一候选改为有限 covariance 或保守
avoidance limit，只从 control/guard data 估计 slope/curvature variance，并保留 exact
matrix-free operator actions 与 pure-signal transfer/window gates。不得恢复 fixed morphology
template。复现入口：

- `3dnet/ps2d_v2_sky_smooth.py`；
- `3dnet/ops_scripts/evaluate_template_free_sky_smooth_projection.py`；
- `3dnet/ops_scripts/run_template_free_sky_smooth_projection_8wide.sh`；
- `runs/template_free_sky_smooth_20260713/`。

## 11. 无模板有限协方差 MAP screen

### 11.1 目标与隔离

有限协方差候选保留自由 sky maps，目标为
$\|F_{\rm control}(BQa-d)\|_2^2+\lambda\sum_m\|a_m/\tau_m\|_2^2$。它没有固定
foreground morphology；`tau_m` 只给谱列 map 的相对有限方差。control 使用 51,431 个
复数 native-$k_\parallel$ index-0 modes，guard 使用 102,862 个互斥 index-1 modes，
science modes 不参与 fit/selection。完整 feature action 的 adjoint error 最大 `1.55e-15`。

### 11.2 Guard-only 网格

T0/T1 等方差 ridge `1e-8/1e-4/1/1e2/1e3/1e4` 的 guard residual 为
`0.0543986/0.0543986/0.0544118/0.0602718/0.148693/0.458009`。低端两点相对差
`2.33e-8`，满足 `1e-6` plateau stop；入选项是近似 flat 的低-ridge 极限。

`tau1/tau0=1/0.3/0.1/0.03` 没有改善 guard。加入 T2 后，四档
`tau2/tau0=1/0.3/0.1/0.03` 的 guard scores 为
`0.0997902/0.0556316/0.0544600/0.0544111`；曲率方差只能在趋零时回到 T0/T1 极限，
没有被 selector 选择。整个选择过程不读取 foreground/EoR truth。

### 11.3 Transfer 与停止门

冻结入选项 T0/T1、`tau=[1,1]`、ridge `1e-8`、12 iter 的 pure-EoR target L2/power ratio
为 `0.059495/0.945330`，最大单 band 误差 `0.059579`。但 observed target L2 为
`9.851e5`，foreground residual target power 是 EoR 的 `1.425e6` 倍。非入选 ridge-100
诊断的 pure 指标为 `0.056190/0.948382`，foreground ratio 仍为 `1.398e6`。

因此停止 control-trained finite-Gaussian MAP subtraction，不运行扩频、noise split 或
cross-power。该结论不排除 likelihood-level nuisance-covariance marginalization，但明确
排除继续调当前 MAP foreground cube 后直接扣除的路线。复现入口：

- `3dnet/ps2d_v2_sky_smooth.py`；
- `3dnet/ops_scripts/evaluate_template_free_sky_smooth_finite_covariance.py`；
- `3dnet/ops_scripts/select_template_free_finite_covariance.py`；
- `3dnet/ops_scripts/run_template_free_sky_smooth_finite_covariance_8wide.sh`；
- `runs/template_free_sky_smooth_finite_covariance_20260713/`。
