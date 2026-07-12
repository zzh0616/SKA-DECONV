# PS2D v2：逐 Fourier mode 选窗与双产品分箱设计

状态：核心实现与 8 频 dirty-EoR 闭环已通过；现有 nuisance estimator 尚未完成 v2
迁移，因此暂不继续其科学参数测试。

## 1. 设计决定

PS2D v2 不再先形成粗二维 bins、再按 bin center 或整-bin规则选择 EoR window。固定流程为：

1. 冻结频率、角尺度、宇宙学、taper、Nyquist 和 UV support；
2. 给每个三维 FFT mode 计算实际 $(k_\perp,|k_\parallel|)$；
3. 在 mode 层应用 low-$k_\parallel$ floor、wedge、supra-horizon buffer 和 UV support；
4. 只把通过窗口的 modes 聚合到 science bins；
5. 同时保留完整诊断平面，但绝不把完整平面与科学窗口混成一个产品。

该窗口完全由观测几何、前景和仪器规则定义，不读取注入 EoR map、phase、amplitude 或
power spectrum。注入 EoR 在窗口中的功率占比只能在窗口冻结后用于覆盖审计。

## 2. 两套产品

### 2.1 `full_ps2d`

`full_ps2d` 覆盖横向 FFT 内切圆和显式保留的径向 modes，用于检查 foreground wedge、
leakage、null test 和估计器异常。它不是默认 science mask。

### 2.2 `window_ps2d`

`window_ps2d` 先逐 mode 通过几何/UV窗口，再聚合。一个 bin 被 wedge 穿过时，只聚合
真正通过的 modes；不把整个 bin 删除，也不把整个 bin 纳入。每个 bin 必须保存
`selected_mode_fraction`，以便识别 partial bins。

以后可报告的 science support 还需进一步与 truth-blind operator/Fisher support 相交：

$$
W_{\rm report}=W_{\rm geometric+UV}\cap W_{\rm response/Fisher}.
$$

几何窗口本身不保证信号在复杂 forward operator 后可识别。

## 3. 坐标与 FFT 约定

输入 cube 轴序固定为 `[frequency, y, x]`。中心频率只用于横向角尺度换算；径向间距由
全部频率对应的共动距离相邻差的均值计算，并记录各间距相对均值的最大偏差。当前 8 频
配置的最大偏差为 0.5071%，因此继续使用窄带均匀 FFT 近似；更宽频段若超过预设容差，
应先冻结重采样方案或改用非均匀 Fourier transform，不能静默沿用均值。

对尺寸为 $(N_f,N_y,N_x)$ 的实 cube，直接使用

$$
k_\parallel=2\pi\left|{\rm fftfreq}(N_f,d=\Delta\chi)\right|,
\qquad
k_\perp=\sqrt{k_x^2+k_y^2}.
$$

正负 $k_\parallel$ 不再重新切成等宽区间，而是合并到实际的非负离散 mode 值。偶数频率
cube 的径向 Nyquist 是否保留必须显式配置；当前为兼容旧审计而选择 `exclude`。

横向只保留内切圆。普通 bins 为左闭右开，最后一个 bin 的右边界显式包含，避免
`np.digitize` 静默丢掉恰好落在最大边界的 modes。

## 4. 分 bin 规则

- `full_ps2d` 的 $k_\perp$ edges 从 0 到横向内切圆上限，当前使用 32 个线性 bins。
- `window_ps2d` 的 $k_\perp$ edges 直接从实际 UV support 下限到上限，当前
  $30$--$2500\lambda$ 对应
  $0.01915394$--$1.59616136\ {\rm Mpc}^{-1}$，同样使用 32 个线性 bins。
- $k_\parallel$ 列直接标记实际 modes，不使用旧 edge-center 标签。当前排除径向 Nyquist 后为
  $0$、$0.10151310$、$0.20302620$、$0.30453930\ {\rm Mpc}^{-1}$。
- edge centers 只可用于显示；每个 populated bin 另存 mode-count-weighted 实际
  $k_\perp$ 与 $k_\parallel$ 坐标。不得用 EoR power 加权坐标。

当前有限 512 像素、32 arcsec source patch 的主模拟窗口为

$$
k_\parallel\ge0.2h\ {\rm Mpc}^{-1},
\qquad
k_\parallel\ge0.23941631k_\perp+0.1h\ {\rm Mpc}^{-1},
$$

并与上述 UV support 相交。finite-patch slope 只适用于本次无 PB、有限输入天空仿真；
真实 SKA 必须使用由 phase centre、beam/full-sky support、LST 和 baseline chromaticity
冻结的窗口，不能直接移植该 slope。

## 5. 功率计算与归一化

当前无噪声兼容基准固定为 global demean、视向 Hann、二维空间 Hann。设三维 taper 为
$w$，其均方为 $U=N^{-1}\sum_nw_n^2$，voxel volume 为
$V_{\rm vox}=\Delta x\Delta y\Delta\chi$，则单个 FFT mode 的 auto-power 定义为

$$
P_{\boldsymbol k}
=\frac{V_{\rm vox}}{NU}
\left|{\rm FFT}\left[w(d-\bar d)\right]_{\boldsymbol k}\right|^2.
$$

实现必须保存 `window_energy`、`power_scale`，并验证

$$
\sum_{\boldsymbol k}P_{\boldsymbol k}
=\frac{V_{\rm vox}}{U}\sum_n\left[w_n(d_n-\bar d)\right]^2
$$

的 Parseval 闭环。未来噪声测试使用相同归一化的 split cross-power

$$
P^{AB}_{\boldsymbol k}
=\frac{V_{\rm vox}}{NU}
\operatorname{Re}\left[F_A({\boldsymbol k})F_B^*({\boldsymbol k})\right],
$$

允许单 mode 或单 bin 为负，不能在聚合前截断为正值。Blackman--Harris 视向 taper 已作为
显式选项实现，但改变 taper 会改变 analysis-contract hash，必须作为独立冻结实验，不能与
当前 Hann 结果混报。

## 6. 每个 band 必须保存的统计量

每个 full/window band 保存：

- 原始 FFT mode count；
- 去掉实数据共轭冗余后的 conjugate-unique mode count；
- mode power 的精确 `sum`；
- mode power 的 `mean`；
- bin 内标准差；
- 矩形 support 与逐模窗口之间的 selected-mode fraction；
- edge center 和实际 mode-count-weighted 坐标。

二维形状比较可使用 band mean。跨 bands 的总功率与 recovered/truth power ratio 必须使用
`sum(mode power)`，不能对各 bin mean 做无权求和；否则稀疏 bin 与包含上千 modes 的 bin
会得到相同权重。

## 7. 8 频闭环结果

标准 dirty-EoR cube 的 v2 结果为：

- Parseval 相对误差 $2.22\times10^{-16}$；
- full 和 science-window 聚合与直接逐 mode 求和的相对误差分别为 0 和
  $2.22\times10^{-16}$；
- science window 保留 31,512 个 FFT modes、15,756 个 conjugate-unique modes；
- 32 个 science bands 非空，其中 30 个完整通过，2 个由 wedge 穿过并按 mode 部分保留；
- science-window/full power 为 `0.10064739015909584`。

最后一个数与旧审计中正确的逐 mode 结果逐位一致。旧严格 `all_modes` 只得到 5.3536%，
是整-bin 丢弃造成的口径损失，不是物理窗口结果。旧坐标下有 4 个 partial bins，其中还混入
UV 边界；新 science edges 从 UV support 起切后，只剩两个真正由 wedge 造成的 partial bins。

机器可读结果位于
`runs/ps2d_v2_mode_first_20260712/8wide_dirty_eor_patch_uv/result.json` 和
`bandpowers.npz`。冻结 layout hash 为
`fe09288426ac1c118605ff53bfae52191ddf8ea2cb5f61e4feb28f8b800b2326`；包含
demean、taper、归一化和布局的 analysis-contract hash 为
`ce60b514464478c8d5543850805cc5f417f2bcae43f192e42adec06381bd8e64`。

## 8. 估计器迁移步骤

1. 用 v2 layout 的 `selected_mode_indices` 和 `selected_mode_bands` 重新构造 science
   signal covariance/projectors；不能先形成 legacy PS2D 再 mask。
2. control/guard modes 使用单独、明确的 full-layout selector，不能与 science selector
   复用含义不明的二维 boolean mask。
3. signal、foreground response、Gaussian unit-band probes、Fisher/window calibration、
   covariance 和最终 evaluator 必须记录并核对同一个 analysis-contract hash。
4. probe 必须在 taper 和 mode selection 之前注入；每个 requested band 的 probe 只激活
   v2 定义中的 modes。
5. 先做 identity estimator 与 pure-EoR closure，再做 foreground nuisance；identity 情况下
   每个 supported band 的 transfer、window localization 和总功率必须闭环。
6. 无噪声 8wide 通过后才冻结迁移并运行 16wide；噪声阶段必须使用独立 split cross-power，
   同时报告 covariance、null tests 和 negative bandpowers，不允许正值截断。
7. `powerspec.py` 和旧 `all_modes` config 仅保留历史复现。任何旧结果若未记录 v2
   analysis-contract hash，都不得与新 science product 合并排名。

## 9. 暂停与验收规则

在 estimator 完成上述迁移前，暂停新的 optimizer、prior、bin-removal 和频率扩展测试。
迁移至少必须通过：

1. 单元测试：原生 $k_\parallel$、Nyquist、右边界、部分 bin、独立 mode count、Parseval、
   auto/cross 同归一化；
2. 数据闭环：8 频 dirty-EoR 逐 mode sum 与 v2 聚合一致到 float64 精度；
3. calibration 闭环：identity operator 的 probe window 为预期的 taper-induced window，且
   signal 与 probe contract hash 完全相同；
4. pure-EoR closure：不允许 estimator 在无 foreground 时产生大于预登记容差的 transfer bias；
5. truth blindness：窗口、support 和停止条件不读取注入 EoR 真值。

只有这些门通过后，才重新评估 nuisance-hardened/QML 路线是否改善复杂 operator 后的
可识别区域。
