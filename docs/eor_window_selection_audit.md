# EoR window 逐模式审计

## 结论

旧 8 频实现按 PS2D bin center 判断
$k_\parallel\ge0.1356\ {\rm Mpc}^{-1}$，误收了实际
$k_\parallel=0.101513\ {\rm Mpc}^{-1}$ 的整层模式。旧中心法显示窗口包含
55.33% 的 active dirty-EoR FFT power；逐 Fourier mode 修正后，历史矩形门槛只包含
10.66%。因此旧 96-bin 口径不能继续作为正式 EoR-window 指标，修正后的矩形支持为
64 个完整 bins。

实践中的 EoR window 不是固定矩形，而是以下信息的交集：

1. beam、有效 sky support 或 horizon 定义的 wedge；
2. 为 supra-horizon leakage 保留的 buffer；
3. 数据和系统项决定的低 $k_\parallel$ floor；
4. 实际 baseline/UV support；
5. 与 EoR truth 无关的 operator/Fisher support。

窗口由前景和仪器定义，而不是按模拟 EoR 功率大小选择。窗口完全可能删除 EoR
功率较大的模式；注入功率占比只能在冻结窗口后作为后验诊断。

## 当前 8wide 数值

- 频率为 117.9--120.7 MHz，中心 119.3 MHz，步长 0.4 MHz；非负离散
  $k_\parallel$ 为 0、0.101513、0.203026、0.304539、0.406052 Mpc$^{-1}$。
- $0.1356\ {\rm Mpc}^{-1}=0.2h\ {\rm Mpc}^{-1}$，所以第一条真正通过 floor 的
  模式是 0.203026 Mpc$^{-1}$。
- 512 x 512、32 arcsec source patch 的角落半径为 3.218 度，对应 flat-sky wedge
  slope 0.239416；常规 horizon slope 为 4.264836。
- 30--2500 lambda 对应
  $0.019154\le k_\perp\le1.596161\ {\rm Mpc}^{-1}$。

按 global demean、3D Hann、横向内切圆和现有 radial-Nyquist exclusion，直接对
$|FFT|^2$ 逐 mode 求和：

| profile | mode 占比 | 逐-mode EoR power | 严格完整-bin power |
| --- | ---: | ---: | ---: |
| 逐模式 0.2h 矩形 floor | 57.1429% | 10.6607% | 10.6607% |
| 有限 patch wedge + 0.2h floor + 0.10h buffer | 8.7588% | 10.6520% | 10.6501% |
| 上式再交 30--2500 lambda | 8.7533% | 10.0647% | 5.3536% |
| horizon wedge + floor + buffer | 0.02556% | 1.6756% | 0 |
| 上式再交 30--2500 lambda | 0.02000% | 1.0883% | 0 |

有限 patch profile 是当前有限输入 sky model 的仿真专用主口径；horizon profile 是
压力测试。真实 SKA 是宽视场相位跟踪阵列，必须换成 beam/full-sky、LST、phase centre
和阵列纬度相关的 horizon line，不能直接沿用 patch slope。

patch+buffer+UV 的最低两个 $k_\perp$ partial bins 各有 88.89% 模式满足边界，并分别
贡献全部 active EoR power 的 4.0981% 和 0.6112%。严格整-bin策略把它们全部删除，才使
占比从 10.0647% 降为 5.3536%。这不是永久删除这两个 bins 的物理依据；正式实现应在
binning 前逐 mode 过滤，或重新设计与 UV/wedge 边界对齐的 bins。horizon profile 在
当前 8 频网格上只有两个 partial bins，没有任何完整 bin。

## 实现要求

- 在 binning 前逐 mode 判定窗口，并保存每个 band 的 selected-mode fraction。
- 整 bin estimator 使用 `all_modes` 时必须单独报告 partial bins；它是保守兼容策略，
  不能代替逐-mode estimator。更好的方案是先切 mode，或增加频率数并重新设计与物理
  边界对齐的 bins。
- 中心频率只用于横向宇宙学换算；径向频率 grid 必须另存首频。旧实现混用两者，使
  当前径向间距偏低 0.584%。
- 保留 radial Nyquist exclusion 作为当前保守兼容约定，但在元数据中明示；该层含
  全 circle dirty-EoR power 的 0.6965%。
- 几何窗口不能代替可识别性判定。最终 science mask 必须是 geometric/UV window 与
  truth-blind response/Fisher support 的交集。

## 文献依据

- PAPER wedge 与 supra-horizon leakage：https://arxiv.org/abs/1301.7099
- wedge slope、低模 floor 和 wedge bias：https://academic.oup.com/mnras/article/456/1/66/1069472
- HERA horizon + 200 ns buffer：https://arxiv.org/abs/2108.02263
- MWA 数据相关低模与 baseline cuts：https://arxiv.org/abs/1911.10216
- SKA 等相位跟踪宽视场阵列的 full-sky wedge：https://arxiv.org/abs/2407.10686
