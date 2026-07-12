# 算子感知、nuisance-hardened 二维功率谱估计器

## 详细实现与验证计划

状态：PS2D v2 identity/pure-EoR 迁移已经通过，但当前 control-only compiled nuisance
projector 没有超过 raw foreground avoidance，故按预登记门槛停止在 8wide 无噪声阶段。
未运行 16wide、noise/split cross-power 或 full-Fisher。结果见
[`ps2d_v2_estimator_validation_results.md`](ps2d_v2_estimator_validation_results.md)。

## 1. 结论摘要

2026-07-12 的逐 Fourier-mode 审计发现，旧实现按二维 bin center 判断窗口，在 8 频
粗网格上错误地把实际 $k_\parallel=0.1015\ {\rm Mpc}^{-1}$ 的模式纳入
$k_\parallel\ge0.1356\ {\rm Mpc}^{-1}$。详细审计见
[`eor_window_selection_audit.md`](eor_window_selection_audit.md)。以下计划中的窗口必须使用
修正后的 mode-level 定义。

重叠最严重的位置是最低的横向波数和低视向波数。当前 floor 为
$k_\parallel=0.1356\ {\rm Mpc}^{-1}=0.2h\ {\rm Mpc}^{-1}$；8 频离散模式中真正通过
门槛的第一层是 $0.2030\ {\rm Mpc}^{-1}$，对应当前 target band 的中心约
$0.2538\ {\rm Mpc}^{-1}$。旧中心约 $0.153\ {\rm Mpc}^{-1}$ 的 band 应视为 guard，
不是 EoR-window band。随着 $k_\parallel$ 增大，可识别区域会逐渐向更高
的 $k_\perp$ 扩展，因此问题不是“整个 EoR window 都不可识别”，而是“传统
EoR window 中包含一部分不可识别或弱可识别模式”。

传统 EoR-window cut 是一个几何上的 foreground-avoidance 规则。复杂、色散的成像
算子作用后，它不能保证前景响应子空间和 EoR 响应子空间仍然正交。因此，新估计器
必须从 Fisher matrix 和 window function 中独立推导可报告区域，不能预设
“落入传统 EoR window 的所有 bin 都可以测量”。

本计划不再把“逐像素、逐相位恢复 EoR 图像”作为首要目标，而是估计有明确
window function 的二维 bandpower。完整链条为：

1. 使用已有前景模板和已有低阶频谱系数；
2. 使用 exact 或 cached forward response；
3. 解析边缘化 foreground nuisance；
4. 使用独立数据 split 的 cross-power 消除加性噪声偏置；
5. 使用 quadratic estimator 和 Fisher/window calibration 估计 bandpower；
6. 明确输出 supported、weak 和 unsupported 模式。

该方法不使用预训练网络，不使用注入 EoR 模板，不固定仿真的 EoR 功率谱形状，也
不在实际拟合或模型选择中使用 oracle foreground coefficients。

## 2. 提出该测试的实验依据

下表中的 EoR 和前景真值只用于拟合完成后的仿真诊断。

| 测试 | 结果 | 含义 |
| --- | ---: | --- |
| hard full64/block8 nuisance projection，$k_\perp32\times k_\parallel4$ | pure-EoR 加权相对 L2 约 0.140 | projection 本身已经移除或混合了可测量的 EoR 功率 |
| 同一 projection | current/bad/oracle 约 0.115/0.097/0.105 | 三者均差于各自 no-correction baseline |
| 更细的 $k_\perp128$ projection | rank 511/512，条件数约 2549 | 稀疏 bin 接近奇异响应组合，并没有因分箱更细而更可识别 |
| no-projection foreground avoidance，8wide | 加权 L2 0.00731；总功率比 1.0036 | 集成功率很好，但不能证明所有二维 bin 都恢复正确 |
| no-projection foreground avoidance，16wide | 加权 L2 0.00636；总功率比 0.9981 | 独立频率规模上复现，但仍有同样限制 |
| 全 PS2D 平面的 truth-only 诊断 | 8wide 为 26/128 bins；16wide 为 87/256 bins 同时通过两个 10% 门槛 | 可恢复性在二维平面上高度不均匀 |

最后一行使用的 truth-only 门槛为

$$
\left|P_{\rm est}/P_{\rm EoR}-1\right|\le 0.1,
\qquad
P_{\rm FG\ mismatch}/P_{\rm EoR}\le 0.1.
$$

这些门槛可以验证一个预先定义的选择规则，但不能用于生成选择规则。使用修正后的
32 arcsec 空间像素和 mode-level floor 后，第一个 target/EoR-window band 附近的
严格可恢复区域约为
$k_\perp=0$ 到 $0.257\ {\rm Mpc}^{-1}$。在更高的 $k_\parallel$ 上，该区域会扩展；
例如 8wide 诊断中，在 $k_\parallel\simeq0.357\ {\rm Mpc}^{-1}$ 附近可达到
$k_\perp=0$ 到 $0.836\ {\rm Mpc}^{-1}$。

这些结果给出两个硬性要求：

- 必须报告传统 EoR window 与独立 response-support mask 的交集；
- Fisher/window response 奇异、不稳定或过宽的 bin 不允许强行 deconvolve，也不允许
  作为已恢复的科学结果报告。

## 3. 范围和非目标

### 3.1 第一阶段目标

第一阶段估计 dirty EoR component 的 cylindrical power spectrum
$P(k_\perp,k_\parallel)$。taper、Fourier convention、宇宙学距离换算、bin edges、
逐 mode 窗口、横向内切圆 mask 和 Nyquist 约定必须与现有 dirty-EoR evaluator 完全
一致。global demean 已去除精确 DC，正的 $k_\parallel$ floor 另行排除零径向层，不能
因一个轴上零模删除整个最低 $k_\perp$ bin。

这样可以先验证 dirty-domain estimator，避免在尚未闭环时声称完成 intrinsic-sky
deconvolution。只有 dirty-domain 测试通过后，才考虑把 imaging operator 放在 sky
band covariance 的两侧，估计 intrinsic-sky bandpower。

### 3.2 允许使用的信息

- 测量得到的 dirty cube，以及独立的 time/visibility splits；
- 当前约 10% 误差量级、无仪器效应的 foreground sky template；
- 已有低阶前景频谱参数化，第一阶段使用 $C_0,C_1,C_2$，宽频段可测试已经有依据的
  cubic 项；
- exact response 或带完整 identity manifest 的 stride4/rank64 cached proxy；
- 从 split difference、off-source data 或有文档的 instrument-noise model 得到的
  thermal-noise covariance；
- 固定的 Fourier geometry 和公认的宇宙学换算。

### 3.3 禁止使用的信息

- 在拟合、停止条件、阈值选择或 bin 选择中使用注入 EoR map、phase、amplitude 或
  power spectrum；
- 在可部署拟合或方法选择中使用 oracle foreground coefficients；
- 使用仿真 EoR 空间结构预训练神经网络；
- 在不同 band 之间固定一个仿真的 EoR covariance shape；
- 根据 recovered/truth ratio 事后挑选“表现好”的 bin。

### 3.4 本计划不试图完成的事情

- 对与 nuisance subspace 重合的模式恢复唯一 EoR 图像；
- 用平滑插值补齐 unsupported modes；
- 仅靠换 Adam、增加 iteration，或者把 image MSE 改写成全 Fourier 平面等权 MSE
  来改善可识别性。

## 4. 为什么只把 loss 移到 Fourier 空间不够

对于 unitary Fourier transform $F$，

$$
\|d-m\|_2^2=\|F(d-m)\|_2^2.
$$

因此，把 image MSE 换成全 Fourier 平面等权 MSE，只是改变表示形式，没有改变反演
问题本身。

本计划的方法与上述改写有三个本质区别：

1. 估计的是 covariance bandpower，而不是 Fourier phase 或 EoR map；
2. foreground mean response 被解析 profile 或 marginalize；
3. requested bandpower 和 measured bandpower 之间的 mixing 会被 calibration，并且
   unsupported combination 会被丢弃。

这可以避免要求前景拟合去复现 EoR 的具体形态，但不能恢复与 foreground nuisance
response 完全共线的信号模式。此类模式必须表现为很小的 Fisher eigenvalue 或很宽的
window function，并被标记为 unsupported。

## 5. 统计模型

### 5.1 数据和前景均值

对数据 split $s$，定义

$$
y_s=d_s-m_{0,s},
$$

其中 $d_s$ 为 dirty data cube，
$m_{0,s}=A_sT(c_0)$ 为当前 foreground template 经过该 split 的响应。

只线性化已经接受的 foreground degrees of freedom：

$$
m_{\rm FG}(c_0+\delta c)
\simeq m_0+J\delta c,
\qquad
J_i=A\left[\frac{\partial T}{\partial c_i}\right]_{c_0}.
$$

$J$ 的列是 canonical compiled Chebyshev responses。当前 cached operator 是逐频率的
空间 low-rank response；频率耦合来自 foreground spectral basis 和三维功率谱变换，
而不是来自一个经过训练的 frequency PCA。

每个 compiled-response 文件必须记录：

- operator type、stride、PCA rank、tile geometry、frequency list、crop 和 dtype；
- source template、coefficients 和 cache metadata 的 hash；
- Chebyshev frequency normalization 和 coefficient ordering；
- 独立 direct-forward probe 的相对误差。

任何 identity mismatch 都必须直接报错，不能自动继续。

### 5.2 分析算子和 bandpower covariance

定义固定的 analysis operator $B_s$，它包含 taper、valid-pixel mask 和确定性的
weighting：

$$
z_s=B_sy_s,\qquad
J_s^{\rm ana}=B_sJ_s,\qquad
N_s^{\rm ana}=B_sN_sB_s^\dagger.
$$

下文公式中的 $y$、$J$ 和 $N$ 均指这些 analysis-space quantities。若两个 split
具有不同 sampling，则各自使用 $B_s$、$J_s$ 和 $N_s$。

残差 covariance 建模为

$$
C(p)=N+\sum_b p_bQ_b,
$$

其中每个 $p_b$ 都是独立的非负 bandpower，不对不同 $p_b$ 之间施加平滑关系。

对于第一阶段的 dirty-domain target，先定义 unit dirty-sky covariance：

$$
S_b^{\rm dirty}=F^\dagger\Pi_bF.
$$

对应的 analysis covariance derivative 为

$$
Q_b=B S_b^{\rm dirty}B^\dagger,
$$

其中 $\Pi_b$ 选择一个 $(k_\perp,k_\parallel)$ bin，并包含 transverse-circle 和
DC masks。归一化必须与现有 powerspec implementation 一致。

对于以后可选的 intrinsic-sky target：

$$
Q_b^{\rm ana}=BA S_bA^\dagger B^\dagger.
$$

只有 dirty-domain 测试通过，并且 adjoint identity 已经验证，才允许启用该扩展。

### 5.3 Restricted nuisance marginalization

对 foreground correction coefficients 使用 flat prior，定义

$$
P_C=
C^{-1}
-C^{-1}J
\left(J^\dagger C^{-1}J\right)^+
J^\dagger C^{-1},
$$

其中 $+$ 表示使用有记录 cutoff 的 SVD pseudoinverse。

restricted negative log likelihood 为

$$
\mathcal L_{\rm REML}(p)
=\frac{1}{2}
\left[
y^\dagger P_Cy
+\log|C|
+\log\left|J^\dagger C^{-1}J\right|
\right].
$$

它在不施加很窄 coefficient prior 的情况下处理前景系数。较宽、且有物理依据的
coefficient bound 可以作为 robustness test，但不是 baseline。

quadratic score 和 Fisher matrix 为

$$
q_b=
\frac{1}{2}
\left[
y^\dagger P_CQ_bP_Cy
-{\rm Tr}(P_CQ_b)
\right],
$$

$$
F_{bb'}=
\frac{1}{2}
{\rm Tr}(P_CQ_bP_CQ_{b'}).
$$

使用 Fisher scoring：

$$
p^{(t+1)}
=
\Pi_{p\ge0}
\left[
p^{(t)}+\alpha_tF^+q
\right],
$$

并使用 restricted likelihood line search，或者用 softplus 保证非负。baseline solver
不是 Adam。小规模 exact problem 使用 SVD-whitened Newton 或 trust-region solve；
完整问题使用 matrix-free Fisher scoring。

$p_b$ 的初值从 unprojected split cross-spectrum 的非负部分得到，并除以 unit-band
response。非正值替换为由观测中正 bandpower 中位数得到的小公共 floor。这样只使用
data scale，不固定频谱形状。必须从 $0.1p^{(0)}$ 和 $10p^{(0)}$ 各重复一次；只有当
三种初值收敛到统计上一致的 windowed bandpower 和相同 support class 时才可报告。
foreground coefficients 被解析 profile，因此没有 optimizer initial value。

### 5.4 Cross-split quadratic estimator

auto-power 存在 noise bias。对于独立 splits，计算

$$
q_b^\times=
{\rm Re}
\left[
(P_1y_1)^\dagger Q_b(P_2y_2)
\right].
$$

normalization 和 band-mixing matrix 使用 signal-only probes 标定；probes 必须经过
相同的 split operators 和 projectors。独立噪声的期望 cross-power 为零。如果两个
split 存在已知 correlated noise，必须显式减去 cross-noise term，并将其不确定性传播
到最终 covariance。

把同一 dirty cube 复制两份不构成独立 split，也不能作为 cross-power 测试。

split definition 必须在查看 science power 前固定：

- 首选：具有匹配 $uv$ coverage 的交错 time blocks；
- 备选：互不重叠的 visibility sets，并为每个 split 记录 response；
- 仅用于仿真工程测试：同一天空加两个独立 thermal-noise realizations。

### 5.5 可选的 nuisance hardening

如果 calibration data 能提供 foreground-residual covariance templates
$Q_\alpha$，可把它们的自由 amplitude 加入 quadratic parameter vector，然后用
Fisher Schur complement 去除其线性响应：

$$
q_p^{\rm hard}
=q_p-F_{p\alpha}F_{\alpha\alpha}^+q_\alpha,
$$

$$
F^{\rm hard}
=F_{pp}
-F_{p\alpha}F_{\alpha\alpha}^+F_{\alpha p}.
$$

允许的 $Q_\alpha$ 只能来自 instrument calibration residuals、
catalog/template perturbations 或独立 observed data，不能从注入 EoR realization
估计。如果 hardening 使 $F^{\rm hard}$ 奇异，对应 combination 必须标记为
unsupported，不能用先验填补。

## 6. 可识别性和最终报告的 window

### 6.1 必须计算的诊断

所有 identifiability diagnostics 必须在与 EoR truth 比较之前完成：

1. 使用相同 noise metric whiten $J$ 和 unit-band response probes；
2. 计算 nuisance subspace 与每个 band-response subspace 之间的 principal angles；
3. 用固定随机种子、与 observation 无关的 Gaussian band probes 标定 transfer/window
   matrix $W$；
4. 计算 normalized Fisher eigensystem，并测试其对 probe count、CG tolerance 和
   SVD cutoff 的稳定性；
5. 输出 row-normalized windows、effective row width、远邻 leakage、retained signal
   response 和 Monte Carlo uncertainty。

最终 estimate 是一组可测量的线性组合：

$$
\widehat p=Mq,
\qquad
W=MF.
$$

minimum-variance normalization 可能产生较宽 window；decorrelated normalization
可能产生较大 error bars。两者都必须发布 $W$。没有检查 window row 时，不能把结果
称为某个直接 bin 的测量。

### 6.2 预先登记的 support gates

第一版实现使用以下工程门槛；这些门槛必须在查看 mixed-signal truth 前固定：

- float64 下只保留
  $\lambda_i/\lambda_{\max}\ge10^{-6}$ 的 normalized Fisher eigenmodes；
- 一个可报告 bin 的 retained signal response 至少为 0.8；
- Monte Carlo transfer uncertainty 小于 10%；
- row-normalized window 至少 50% 位于 nominal bin 内；
- nominal bin 及其直接相邻 bins 之外的 leakage 不超过 25%；
- probe count 加倍，并把 SVD cutoff 改变十倍后，estimate 与 support status 的变化
  不超过 10%。

这些是 numerical/reporting gates，不是 astrophysical priors。可以根据 null test 和
pure-response engineering test 修订，但不能通过最大化与 injected EoR truth 的一致性
来修订。

每个 bin 必须得到一个状态：

- supported：全部门槛通过；
- weak：response 有限，但至少一个 resolution 或 stability gate 未通过；
- unsupported：response 奇异、近奇异，或者 numerical closure 失败。

最后再把 conventional EoR window 与该 mask 相交。输出图必须区分四类区域：
outside-window、window-and-supported、window-and-weak 和
window-and-unsupported。

## 7. 建议代码结构

当前仓库是早期 flat-script 结构。新 estimator 应作为独立 package 加入，保持已有
UNet 和 polynomial scripts 行为不变：

~~~text
ps2d_estimator/
  __init__.py
  config.py
  geometry.py
  operator.py
  compiled_nuisance.py
  covariance.py
  qml.py
  cross_power.py
  identifiability.py
  diagnostics.py
  io.py
scripts/
  run_nuisance_hardened_ps2d.py
  evaluate_nh_ps2d_simulation.py
configs/
  nh_ps2d_toy.toml
  nh_ps2d_8wide.toml
  nh_ps2d_16wide.toml
tests/
  test_geometry.py
  test_compiled_nuisance.py
  test_qml_dense.py
  test_qml_matrix_free.py
  test_cross_power.py
  test_identifiability.py
~~~

各模块职责：

- config.py：typed configuration、path resolution、deterministic seeds 和序列化
  run manifest；
- geometry.py：宇宙学换算、taper、FFT normalization、bin indices、circle mask、
  DC mask 和 conventional EoR-window mask；
- operator.py：identity、exact response 和 cached stride4/rank64 response 的统一
  apply/adjoint protocol；
- compiled_nuisance.py：加载并验证 canonical $J$，实现 $J$ 和 $J^\dagger$，构造
  小型 nuisance Gram matrix，并提供 principal-angle probes；
- covariance.py：matrix-free $N$、$Q_b$、$C$ 和可选 calibration-residual
  covariance actions；
- qml.py：restricted projector、CG/block-CG、quadratic scores、trace estimates、
  Fisher scoring 和 window normalization；
- cross_power.py：split validation、cross-quadratic scores、split nulls 和
  signal-only transfer calibration；
- identifiability.py：Fisher SVD、principal angles、support gates 和 window
  diagnostics；
- diagnostics.py：truth-free run diagnostics，以及必须显式 allow-truth 才能运行的
  simulation-only evaluator；
- io.py：FITS/NPZ loading、hash、atomic outputs 和 schema validation。

### 7.1 现有实现的迁移起点

fg_rmw workspace 中现有 compiled-response prototype 已经为以下行为提供测试参考：

- canonical v3 compiled nuisance response 的加载；
- 小矩阵 nuisance projection；
- Monte Carlo transfer/window calibration；
- operator-bank identity checks；
- 与修正后二维 powerspec binning 的 parity。

实现时应把这些行为和测试迁入本 package，不能让 SKA-DECONV 在运行时 import 相邻
workspace。cached response arrays、FITS cubes 和 truth products 都作为外部数据管理，
不得提交到本仓库。

## 8. 具体实现步骤

### Step 1：冻结 geometry 和 data contracts

规定数组布局：

- dirty data：[split, frequency, y, x]，float64；
- compiled nuisance：[parameter, frequency, y, x]，float64；
- 可选 sky probes：[probe, frequency, y, x]；
- band powers：[k_perp_bin, k_parallel_bin]。

移植修正后的二维 index validity check。不能先 flatten 二维 indices，再只检查一维
范围。优先从 FITS WCS 读取 32 arcsec pixel size；配置与 WCS 冲突时直接报错。

使用 dimensions、frequencies、WCS spacing、taper、cosmology、bin edges 和 masks
生成 geometry hash。每个 design、probe bank、estimate 和 evaluation product 都必须
携带该 hash。

### Step 2：实现并测试 operator adapters

统一最小接口：

~~~python
class LinearOperator:
    def apply(self, sky):
        ...

    def adjoint(self, dirty):
        ...

    def identity(self):
        ...
~~~

实现：

1. IdentityDirtyOperator：第一阶段 dirty-bandpower target；
2. CompiledForegroundOperator：加载 $m_0$ 和 $J$；
3. CachedPCAOperator：封装逐频率 stride4/rank64 cache；
4. ExactResponseOperator：用于小规模 closure probes。

必须通过：

- float64 linearity error 小于 $10^{-12}$；
- adjoint inner-product error 小于 $10^{-10}$；
- 逐频率记录 cached-versus-direct response error；
- frequency、crop、rank、stride 或 hash 不一致时 hard failure。

### Step 3：实现 Fourier band actions

预计算一个 integer band-index cube，不为每个 band 构造 dense matrix。
$S_b^{\rm dirty}$ 的 apply 过程为 FFT、选择一个 band、inverse FFT 和 normalization；
$Q_b$ 还需要在其前后分别应用 $B^\dagger$ 和 $B$。GPU 上批处理多个 band/probe。

必须测试：

- 与现有 PS2D implementation 的 exact parity；
- Parseval closure；
- $k_\parallel$ 最高边界没有 bin wraparound；
- real cube 的 Hermitian handling 正确；
- direct power 与 quadratic form $x^\dagger Q_bx$ 一致。

### Step 4：实现 nuisance projection

先使用 $C=N$ 和 dense small nuisance algebra。验证

$$
\|P_CJ\|/\|J\|<10^{-10},
\qquad
\|P_CCP_C-P_C\|/\|P_C\|<10^{-10}.
$$

然后用 preconditioned CG 支持一般 matrix-free $C^{-1}$。一个 outer bandpower
iteration 内，block solve $C^{-1}J$ 并复用结果。小矩阵
$J^\dagger C^{-1}J$ 必须使用 SVD，并记录所有 singular values。

### Step 5：实现 score 和 Fisher calibration

提供两个 backend：

- dense_exact：tiny cubes、explicit matrices 和 finite-difference checks；
- matrix_free：production cubes。

matrix-free traces 使用固定 Rademacher probes，并在不同 iteration 使用 common random
numbers。band-mixing matrix 优先使用 direct Monte Carlo response calibration，因为
它同时检查 taper、projection、binning 和 normalization 的完整链条。

full-scale REML log determinant 需要 stochastic Lanczos quadrature。第一版
score/Fisher prototype 不依赖它：先实现 Fisher scoring，用 dense REML value 和
small-cube finite differences 验证方程。只有 line-search stability 确实需要时，再加入
stochastic log determinant。

### Step 6：加入 split cross-power

输入必须是两个具有独立 identity 的 splits，并保存各自 visibility/time selection
hash。如果 sampling 不同，为每个 split 构造自己的 operator。

输出：

- split-1 auto spectrum；
- split-2 auto spectrum；
- cross spectrum；
- half-difference null spectrum；
- 可选 even-odd 和 first-half/second-half consistency spectra。

science estimate 使用 cross spectrum；auto spectra 只作诊断。

### Step 7：输出 identifiability products

NPZ 至少写出：

- raw 和 normalized Fisher matrices；
- Fisher eigenvalues/eigenvectors；
- raw 和 row-normalized window matrices；
- principal-angle summaries；
- transfer mean 和 Monte Carlo standard error；
- supported/weak/unsupported masks；
- conventional EoR-window mask 及其交集；
- bandpower estimates、covariance 和 error bars。

JSON summary 记录 hashes、thresholds、CG convergence、probe convergence 和各
support class 的数量。PS2D 图必须覆盖显示 support status，不能把 unsupported bins
隐藏后再画成完整恢复。

### Step 8：严格分离 estimation 和 truth evaluation

estimator command 不得接受 foreground truth 或 EoR truth 参数。单独的 simulation
evaluator 可以读取 truth，但每个输出指标都必须标记为 postfit_truth_diagnostic。

任何 truth-derived metric 都不能进入：

- optimizer；
- early stopping；
- hyperparameter ranking；
- SVD cutoff selection；
- nuisance rank selection；
- support mask。

### Step 9：提供可复现命令

完成实现后的目标接口：

~~~bash
python -m pytest -q

python scripts/run_nuisance_hardened_ps2d.py \
  --config configs/nh_ps2d_8wide.toml \
  --output runs/nh_ps2d_8wide

python scripts/evaluate_nh_ps2d_simulation.py \
  --estimate runs/nh_ps2d_8wide/estimate.npz \
  --truth-manifest simulation_truth.toml \
  --output runs/nh_ps2d_8wide/truth_diagnostic
~~~

run manifest 必须包含完整 resolved configuration、git commit、environment versions、
input hashes、random seeds 和 operator identities。

## 9. 验证矩阵

### Phase A：代数单元测试

使用 explicit $4\times8\times8$ 或 $8\times16\times16$ cube：

1. 无 nuisance、white noise、只有一个非零 band；
2. 多个相互独立 bands；
3. nuisance subspace 与 signal 正交；
4. 一个 nuisance column 与某个 signal mode 完全相同；
5. 用受控 principal angles 构造近共线 nuisance/signal；
6. 两个具有独立 noise 的 splits。

期望结果：

- supported modes 的 bandpower 在 Monte Carlo uncertainty 内无偏；
- 足够大 ensemble 上，nominal 68% interval coverage 与 68% 相差不超过 5 个百分点；
- cross-noise power 的均值为零；
- exact-degeneracy mode 被标记为 unsupported，而不是被“恢复”；
- dense 和 matrix-free 结果在 $10^{-8}$ relative precision 内一致。

### Phase B：operator closure

使用 pure unit-band Gaussian probes，不加入 foreground 或 injected EoR：

1. identity dirty operator；
2. exact small response；
3. cached stride4/rank64 response；
4. 未参与 PCA cache 构建的 independent exact probes。

通过条件：

- supported bins 的 transfer closure 在 0.9 到 1.1；
- probe 数量加倍后 support classification 稳定；
- 不出现无法解释的 frequency-local discontinuity；
- absolute $k_\perp$ 和 power units 使用修正后的 32 arcsec WCS。

### Phase C：canonical 8wide 测试

使用 central 256、canonical 8wide frequencies、current foreground reference 和
full64 compiled response。

运行以下 estimators：

| ID | Estimator |
| --- | --- |
| B0 | no-correction foreground-avoidance residual PS2D |
| B1 | 现有 hard nuisance projection 加 transfer deconvolution |
| B2 | image-domain coefficient fit |
| B3 | uniform full Fourier-plane coefficient fit |
| B4 | nuisance-marginalized auto-QML |
| B5 | nuisance-marginalized cross-QML |
| B6 | cross-QML 加允许的 calibration-residual hardening |

B2 和 B3 必须在 numerical tolerance 内一致。这是 Parseval control，不是两种独立
方法。

评估三种 foreground reference：

- current deployable template；
- 一个较宽且非 oracle 的 perturbation，例如已有 $C_0$ smooth-relative $10^{-6}$
  stress case；
- oracle 只在方法和所有阈值冻结后，作为 simulation-only stationarity/closure
  diagnostic。

oracle run 不属于 deployable estimator，不能用来调 threshold 或选择 hyperparameter。

### Phase D：独立 16wide 复现

冻结 Phase C 的全部 method choices，然后在 16wide 上运行，不能针对 truth 重新调参。
frequency geometry 改变后，可以从 operator/noise response 重算 support mask，但所有
numerical gates 保持不变。

晋级条件：

- supported-bin bias 和 coverage 不退化；
- support topology 定性一致；
- 如果当前退化确实是物理问题，应独立复现“第一个 EoR band 与 low-$k_\perp$
  weak/unsupported 区重叠”；
- 两个 spatial tile masks 结果稳定。

### Phase E：observation-level split 测试

加入 realistic thermal noise 和 independent time/visibility splits。

必须完成的 null tests：

- split half-difference 在 propagated errors 内与零一致；
- 交换 split labels 后 cross-power 在误差内不变；
- foreground-only simulation 在 supported bins 内没有显著 EoR excess；
- noise-only simulation 的 mean cross-power 为零；
- time 和 baseline partitions 的 jackknife spectra 通过预先登记的检验。

### Phase F：可选 32-frequency 和 intrinsic-sky 测试

只有 Phase A-E 全部通过才继续。32-frequency foreground basis 可使用已经有依据的
cubic spectral term，但不能加入 EoR spectral training。

只有以下条件满足后，才测试 intrinsic-sky
$Q_b=BA S_bA^\dagger B^\dagger$：

- adjoint test 通过；
- exact-versus-cached signal transfer 已量化；
- dirty-domain cross-QML 通过；
- final covariance 包含 operator sensitivity。

## 10. 科学验收标准

总 EoR-window power ratio 接近 1 不能单独构成成功。必须同时满足：

1. injected truth 不进入 fitting、selection 或 support classification；
2. 每个声称可测的 combination 都有 pure-EoR transfer calibration；
3. foreground-only 和 noise-only nulls 通过；
4. supported bins 的 bias 小于 10%，或者小于其预先声明的 statistical uncertainty，
   二者取较大者；
5. reported confidence intervals 具有合理 ensemble coverage；
6. 改善 perturbed foreground case 时，不能把 current closure case 明显推坏；
7. 方法冻结后的 oracle simulation-only closure 不得显示系统吸收 EoR；
8. 16wide frozen-method replication 通过；
9. 每个 claimed band 都发布 window row 和与其他 bands 的 covariance；
10. weak 和 unsupported EoR-window bins 在图和表中保持显式 mask。

oracle closure 只用于冻结方法后的 pass/fail validation，不能用于迭代调参。如果失败，
只能修改方法并从新的预登记测试重新开始，不能根据 oracle 指标选择一个已有
hyperparameter row。

新方法只有在满足以下至少一项时，才替代 no-projection baseline：

- 在不引入 bias 的情况下扩大 supported EoR-window region；
- 在不显著损失 precision 的情况下，为相同区域提供经过 calibration 的 uncertainty
  和 window function。

否则继续把 foreground avoidance 作为 science baseline，并仅把新 estimator 作为
identifiability diagnostic。

## 11. 数值策略和计算预算

operator 和 estimator algebra 使用 PyTorch float64。只有 float64 reference 证明 Fisher
eigenvalues、support classification 和 bandpower 不变后，才允许尝试 mixed precision。

production operations 必须 matrix free：

- FFT masks 实现 $S_b$；
- cached response apply/adjoint 实现可选 $A$ 和 $A^\dagger$；
- block-CG 为 nuisance columns 和 probe batches 求解 $C^{-1}$；
- Hutchinson probes 估计 traces；
- fixed-seed band probes 标定 transfer/windows；
- SVD 只作用于 nuisance-sized 和 band-sized matrices。

初始规划量级如下，正式运行后必须用 benchmark 替换：

| 阶段 | 预计计算量 |
| --- | --- |
| dense algebra/unit tests | CPU 秒级 |
| 8wide、central 256、16-32 probes | 约 10-30 GPU 分钟 |
| 16wide frozen replication | 约 30-90 GPU 分钟 |
| 32-frequency extension | 约 1-3 GPU 小时 |

实际耗时取决于 CG iterations 和 cache locality。每次远程启动前都必须实时检查 GPU
utilization、free memory、host memory、disk 和 cache 是否存在，不能写死某张 GPU
长期可用。

计算上的 early stopping 只使用：

- CG relative residual；
- score norm；
- relative bandpower step；
- likelihood/line-search acceptance；
- probe-convergence stability；
- support mask 是否重复稳定。

不得使用 EoR truth agreement 作为停止条件。

## 12. 失败结果应如何解释

该测试可能得到三种都有科学意义的结果：

1. supported-window recovery：cross-QML 在比 foreground avoidance 更大的区域恢复
   calibrated combinations；
2. honest partial recovery：可恢复区域仍有限，但 Fisher/window products 清楚解释
   原因并给出有效 uncertainty；
3. fundamental overlap：nuisance hardening 后，传统 EoR window 的较大部分仍为
   unsupported。这给出当前数据、foreground degrees of freedom 和 operator 在不增加
   新观测信息时的可识别上限。

第三种结果不是 optimizer failure。它意味着需要更多频率、独立 baselines/times、更好
calibration，或者有充分依据的外部 foreground constraint。没有预训练、从零拟合的 CNN、
更多 hidden layers 或另一种 image loss 都不能创造 measurement 中不存在的信息。

## 13. 2026-07-12 无噪声首测状态

第一版无噪声 screening 已在 fg_rmw/3dnet 中实现并运行。由于完整
$B S_b B^\dagger$ covariance 的 matrix-free solve 成本较高，首测先使用
tapered-Whittle diagonal-covariance approximation，但保留以下计划核心：

- 每个 active PS2D band 使用独立自由 variance；
- compiled C0/block8 foreground response 作为 mean nuisance 解析 profile；
- 使用 restricted likelihood 和 Fisher scoring，不使用 Adam；
- support 由 Fisher window、analytic transfer、principal overlap 和三档初值稳定性
  决定；
- truth 只用于构造无噪声 synthetic observation 和 post-fit diagnostic。

实现通过了 explicit dense projector、score/Fisher、finite-difference REML gradient、
Fourier parity、exact degeneracy，以及 0.1、1、10 三档 bandpower 初值收敛测试。

canonical 8wide、central-256、full64/block8、$k_\perp32\times k_\parallel4$
结果未通过晋级。下表和本节的 96-bin 数字来自后来确认有误的 legacy bin-center mask，
只保留为历史诊断，不能作为正式窗口结果：

| 输入/reference | raw L2 | raw power ratio | REML L2 | REML power ratio |
| --- | ---: | ---: | ---: | ---: |
| pure EoR | 0 | 1 | 15.39275 | 12.47797 |
| current | 0.007308 | 1.003562 | 0.109954 | 0.974609 |
| C0 1e-6 | 0.053729 | 0.974610 | 0.161834 | 0.935351 |
| oracle | 0.026245 | 1.025961 | 0.138939 | 0.953800 |

pure-EoR 最大失败位于
$(k_\perp,k_\parallel)=(0.03215,0.15316)\ {\rm Mpc}^{-1}$，即最低
$k_\perp$ 和旧中心法误收的 guard band，REML/truth 为 16.78。下一条真正进入
mode-level window 的 $k_\parallel=0.25527\ {\rm Mpc}^{-1}$ low-$k_\perp$ band
仍为 2.13。

这不是初值或 iteration 问题：三档初值均在 14--21 iterations 收敛，p95
bandpower spread 不超过 $7.82\times10^{-6}$。更关键的是，truth-free NLL 反而偏好
deliberately perturbed C0 1e-6 reference；current、perturbed 和 oracle 拟合出的
nuisance vectors cosine 为 0.9980--0.9999。预登记 support mask 保留 current 的
92/96 EoR-window bins 后，REML L2 仍为 0.2389，差于相同区域 raw 的 0.0415。

按本计划的停止规则：

- 不晋级 16wide；
- 暂不加入 thermal noise 或 split cross-power；
- 不根据 truth 事后提高 0.8 retained-response threshold；
- foreground avoidance 继续作为无噪声 baseline。

本结果拒绝的是 tapered-Whittle diagonal-covariance candidate，不单独排除完整
matrix-free $B S_b B^\dagger$ estimator。但后者若继续，必须先提出不依赖 truth 的
objective/support 改进，解释并解决 bad-reference NLL ordering 和 pure-EoR support
failure；仅增加计算量不能视为充分理由。

### 13.1 control/guard/target 严格隔离测试

为检查上一轮失败是否来自“用同一批 Fourier modes 同时拟合 nuisance 和评估
bandpower”，又运行了一次 causal target-held-out 无噪声测试。区域在运行前固定为：

- 只用 $k_\parallel$ index 0（中心 $0.05105\ {\rm Mpc}^{-1}$）拟合；
- index 1（$0.15316\ {\rm Mpc}^{-1}$）作为 guard，不拟合也不作为目标；
- indices 2、3（$0.25527$、$0.35738\ {\rm Mpc}^{-1}$）作为 64-bin 主目标区；
- 系数冻结后才在目标区计算 corrected residual PS2D。

隔离单元测试把全部非训练 Fourier rows 加入大幅随机扰动，拟合出的 64 个系数保持
逐 bit 相同。因而下表中的目标结果不存在 target-to-fit data leakage。

| 输入/reference | 主区 raw L2 / ratio | 主区 corrected L2 / ratio | 去掉最低 $k_\perp$ 后 corrected L2 / ratio |
| --- | ---: | ---: | ---: |
| pure EoR | $0/1$ | $2.943/3.202$ | $0.0271/0.9869$ |
| current | $0.04185/1.03661$ | $0.35402/1.27310$ | $0.00262/1.00018$ |
| C0 $10^{-6}$ | $0.04903/1.04264$ | $0.35869/1.27917$ | $0.00614/0.99880$ |
| oracle | $0.04610/1.03995$ | $0.35269/1.27266$ | $0.00263/1.00120$ |

主 64-bin 区域的五项预登记验收全部失败。失败集中在
$k_\perp=0.03215\ {\rm Mpc}^{-1}$、$k_\parallel=0.25527/0.35738\ {\rm Mpc}^{-1}$
两个 bins；pure-EoR recovered/truth 分别为 4.04 和 2.31。去掉这两个 bin 后，剩余
62 bins 的五项门槛全部通过，但这两个 bin 含主目标区 78.08\% 的 injected-EoR
真值功率，62-bin 子区只保留 21.92\%。因此它只能作为严格 held-out 的局部诊断，不能
根据 post-fit truth 升级为新的 science support 定义。

三档 variance 初值的 nuisance-coefficient spread 最大为 $3.07\times10^{-5}$，而训练区
truth-free NLL 仍偏好 deliberately perturbed C0 reference。结论是：把拟合限制在
低-$k_\parallel$ control band 并不能解决最低-$k_\perp$ 目标模式的外推/可识别性问题。
主门失败后，不运行 16wide、thermal-noise 或 split cross-power。

### 13.2 signal-only response window 对低 $k_\perp$ bins 的判定

上一节的 62-bin 结果只通过了聚合恢复检查，不能证明每个 bin 都有局域 response。为避免
按 injected-EoR 功率大小事后挑 bin，又对冻结的 target-held-out estimator 运行了
signal-only window calibration：对全部 128 个 source bands 分别注入 Gaussian unit-band
probes，经过相同 global demean、3D Hann taper、PS2D binning 和 frozen correction，测量
64 个 target outputs 的 response matrix。probes 不读取 injected EoR realization，也不
固定 EoR bandpower shape。

使用预先定义的 self transfer 0.9--1.1、self fraction 至少 0.5、far leakage 不超过
0.25、MC relative SE 不超过 0.1 门槛，current reference 的 16 和 32 probes 得到逐 bin
完全相同的 59/64 support topology。C0 $10^{-6}$ 和 oracle 的 16-probe robustness test
也得到同一 topology。失败的 5 个 bins 为：

| $k_\perp$ | $k_\parallel$ | self transfer | self fraction | far leakage | control leakage / raw self |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.03215 | 0.25527 | 0.99449 | 0.000963 | 0.99831 | 610.21 |
| 0.03215 | 0.35738 | 0.99970 | 0.007159 | 0.98942 | 80.56 |
| 0.09645 | 0.25527 | 0.99940 | 0.05912 | 0.89282 | 8.77 |
| 0.09645 | 0.35738 | 0.99996 | 0.41857 | 0.39047 | 0.520 |
| 0.16076 | 0.25527 | 0.99963 | 0.37105 | 0.31924 | 0.462 |

最低两个 $k_\perp$ bins 各有 90 个 real Fourier rows，而 local compiled design rank 为
64。它们的 signal 本身没有被明显吸收；失败来自 control modes 经过 coefficient fit 后以
巨大增益进入 target output。因此这两个 bins 应从本 correction estimator 的 support 中
排除，但不能从所有 estimator 的 geometric EoR window 中永久删除。未经 correction 的
raw windows 在二者的 self fraction 为 0.542/0.660，control leakage 仅
0.00405/0.000269。

旧配置使用 $k_\parallel\ge0.1356\ {\rm Mpc}^{-1}$、wedge slope 0，因此包含二者。
标准 wedge 边界随 $k_\perp$ 增长，而不是按 EoR 功率大小切除。119.3 MHz 下常规
flat-sky horizon slope 约为 4.265；若采用保守 horizon profile，8wide target 的
geometric window 只剩部分最低 $k_\perp$ modes，而 correction support 又排除相应
整 bins，严格交集为空。不过当前输入 sky model 只有 4.55 度有限图块；仿真专用主
profile 应使用该 source support 的 wedge、supra-horizon buffer 和 30--2500 lambda UV
范围，horizon profile 只作为压力测试。真实 SKA 观测则必须使用 full-sky phase-tracking
horizon line。slope-0 mask 只能称为简化 evaluation window。

逐-mode 审计还显示，patch+buffer+UV 物理边界内含 10.0647% 的 active dirty-EoR
power，但当前粗网格的严格 `all_modes` 整-bin策略只保留 5.3536%。最低两个
$k_\perp$ partial bins 各有 88.89% 模式满足窗口，不能因少数跨界模式而永久删除。
正式 quadratic estimator 必须在 band aggregation 前应用 mode mask，或重分箱使边界
对齐；在此之前，`all_modes` 只作为保守兼容口径。horizon stress profile 没有完整 bin。

response-derived 59-bin mask 的生成不读取 EoR truth。后验影响评估显示它只保留 64-bin
target injected-EoR 功率的 0.906\%；这个比例不是排除依据，但说明本 correction 的可用
区域在当前 simulation 中几乎没有科学价值。因此仍不晋级 16wide 或 noise。最终产品必须
分别发布 geometric EoR window、operator-support mask 和二者交集。

### 13.3 PS2D v2 迁移后的最终 screen

mode-first estimator 迁移随后完成。完整 calibration source basis 覆盖全部 524,288 个
FFT modes；identity transfer、NumPy/Torch parity 和真实 8 频 pure-EoR 均闭环到 float64
精度。Hann taper 的解析 source window 由 64 probes/source 独立复现，证明 v2 契约、
source normalization 和 transfer algebra 本身没有问题。

在相同冻结契约下，control-only compiled nuisance projector 的 32/64-probe 稳定交集为
17/32 science bands。current reference 在该交集上的 raw L2/power ratio 为
0.00896/1.00851，row-normalized corrected 为 0.03659/0.98202；pure-EoR 为
0.04271/0.97630。C0-$10^{-6}$ 和 oracle 同样被 correction 恶化。三档弱 ridge
$10^{-6}$、$10^{-5}$、$10^{-4}$ 也没有超过 raw baseline。

因此本计划的当前 projector 在 8wide 无噪声 gate 即被拒绝。没有为该失败候选继续构造
full Fisher，也没有运行 16wide、thermal noise 或 split cross-power。该停止决定以及
完整 source-window 数值见
[`ps2d_v2_estimator_validation_results.md`](ps2d_v2_estimator_validation_results.md)。

## 14. 参考文献

- Liu and Tegmark, A Method for 21 cm Power Spectrum Estimation in the Presence
  of Foregrounds: https://arxiv.org/abs/1103.0281
- Dillon et al., A Fast Method for Power Spectrum and Foreground Analysis for
  21 cm Cosmology: https://arxiv.org/abs/1211.2232
- Alsing and Wandelt, Nuisance Hardened Data Compression for Fast Likelihood-Free
  Inference: https://arxiv.org/abs/1903.01473
- Sailer et al., Foreground-Immune CMB Lensing with Shear-Only Reconstruction:
  https://arxiv.org/abs/2007.04325
