このファイル（AGENTS.md）は、コーディングエージェント用のドキュメントです。
ユーザーからの指示（「まとめておいて」「覚えておいて」など）に応じて、今後のためにメモして残しておいたほうがいいことや作業ログなどをまとめること。適宜内容を整理して更新すること。

---

## リポジトリ構造

### ルートレベルファイル（主要）
- **README.md**: プロジェクト概要・使用例（`subhalo_properties()` の基本的な呼び出し）
- **requirements.txt**: numpy >= 1.20, scipy >= 1.6, tqdm >= 4.60
- **sashimi_c.py**: **主モジュール**（2170行）。サブハロー物性計算エンジン。
- **radial_pdf_evolution_in_units_of_dynamical_time_nfw_cached_phase_tables.py**: **軌道進化PDF専用モジュール**（1149行）。NFW軌道の時間発展を計算。
- **test_sashimi.py**: sashimi_c の unittest 群（サブハロー質量関数、boost因子等）
- **test_r_dependent.py**: r依存物性計算の unittest 群（Einasto プロファイル比較含む）
- **test_rpdf_integration.py**: 軌道進化PDF統合テスト（r依存の q 値検証）
- **boost_iteration.py**: 暗黒物質annihilation boost因子のパラメータスイープスクリプト

### Jupyter ノートブック（実験/開発用）
- **sample.ipynb**: sashimi_c の最小利用例
- **dev.ipynb**: 開発/デバッグ用ノートブック（変数確認、プロット試験など）
- **dev_r_dependent.ipynb**: r依存計算の試験的検証
- **test_sashimi.ipynb**: 対話的テスト
- その他多数（galpy.ipynb, edgeworth.ipynb, etc）

---

## モジュール詳細

### 1. sashimi_c.py（主エンジン）

**目的**: CDM フレームワーク内でサブハロー物性（質量分布、半径依存分布、tidal mass loss等）を計算。

**主要クラス**（継承階層）:

```
units_and_constants
  ├─ cosmology
  │   └─ halo_model
  │       └─ subhalo_mass_evolution
  │           └─ subhalo_properties ★
  │               └─ subhalo_observables
  └─ time_redshift_evolve
```

**核となるクラス**:
- `units_and_constants`: SI/CGS/天文単位系の定義（Mpc, km/s, Msun等）
- `cosmology`: ΛCDM パラメータ（Ωm, Ωλ, h 等）＆ Hubble パラメータ、成長因子
- `halo_model`: 分散（σM）、濃度（c-M-z 関係）、N-body 初期条件の統計モデル
- **`subhalo_properties`** ★: **メインクラス**
  - `subhalo_properties_calc()`: フル計算（質量関数、密度分布、生存判定等）
  - `subhalo_properties_r_dependence_calc()`: r依存計算（軌道平均分布）
  - 内部で `_p_q_orbit_evolved_from_racc()` を使用
- `subhalo_observables`: 観測的推定量（質量関数、boost因子、衛星銀河数等）

**重要な関数**（トップレベル）:
- `_cosmic_time_seconds(z, *, n=4096, a_min=1e-4)`: スケール因子 a(z) から宇宙時間 t に変換
- **`_p_q_orbit_evolved_from_racc(..., el_mode='sample_Rc_eta')`** ⭐
  - 軌道進化PDF の計算（初期条件から τ までの半径分布）
  - `el_mode='sample_Rc_eta'`: E,Lを確率サンプル（**推奨**）
  - `el_mode='fixed_Ecirc_at_Rvir'`: E固定、L確率（旧）
  - 内部で `radial_pdf_evolution...py` の `build_orbit_cache()` と `evolve_r_from_cache()` を利用

**Tidal Mass Loss エンジン**:
- `subhalo_mass_stripped()`: 質量削減の計算（4つのアルゴリズム可選）
  - `"odeint"`: 微分方程式の数値積分
  - `"pert0"`, `"pert1"`, `"pert2_shanks"` (デフォルト): 摂動展開

---

### 2. radial_pdf_evolution_in_units_of_dynamical_time_nfw_cached_phase_tables.py（軌道進化）

**目的**: NFW 軌道の時間進化と半径分布関数を高速に計算。位相テーブルキャッシング機構を実装。

**ワークフロー**:
1. `build_halo()`: 与えられた (M, z, c) から NFW パラメータ (Rvir, rs, Vs, ρs) を構築
2. `sample_EL_from_eta_Rc_placed_at_Rvir()` ⭐: 初期条件サンプリング
   - `Rc ~ U(0.6, 1.0) * Rvir`, `eta ~ Beta(2.05, C1+1)`
   - E = E_circ(Rc), L = eta * Lc(Rc) を計算
   - r=Rvir を通過できる軌道のみ採択（リジェクトサンプル）
3. `build_orbit_cache()`: 各軌道の位相テーブルを precompute
   - 周期 Tr, 離心率, 近点/遠点 rp, ra, 各 r での tau(r)
4. `evolve_r_from_cache()`: t → r(t) を高速に復元
5. `radial_pdf()`: r の確率分布関数を作成（ヒストグラム）
6. 複数の時刻 t で (2)-(5) を繰り返し、f(r,t) の時間進化を追跡

**主要関数**:

| 関数名 | 説明 |
|--------|------|
| `fNFW(c)` | f(c) = ln(1+c) - c/(1+c) |
| `build_halo(M, z, c, mass_def="200c")` | NFW パラメータ構築 |
| `Phi_NFW(r, rs, Vs2)` | ポテンシャル Φ(r) |
| `dPhi_dr_NFW(r, rs, Vs2)` | dΦ/dr |
| `sample_EL_from_eta_Rc(...)` | η, Rc から (E,L) をサンプル（アウトフォール） |
| `sample_EL_infall_shell_at_Rvir(...)` | 外側からのシェルフォール |
| `sample_EL_from_eta_Rc_placed_at_Rvir(...)` ⭐ | **推奨**: r=Rvir 配置＋通過条件リジェクト |
| `time_averaged_radius_batch_halo(E, L, halo, ...)` | 時間平均半径（複数軌道） |
| `radial_pdf(r, nbins=30, rmin=None, rmax=None)` | 半径ヒストグラムの生成 |
| `build_orbit_cache(E, L, r0, halo, *, n_theta=512)` | 位相テーブル構築 |
| `evolve_r_from_cache(cache, t, *, sign_mode="in")` | t から r(t) を復元 |
| `t_dyn_from_cache(cache, *, method="median_Tr")` | 動力学時間を決定 |
| `main()` | スタンドアローン実行（SASHIMI_FAST=1 対応） |

**特記事項**:
- τ=0（初期条件）では r がほぼ δ(Rvir) なので、プロット時に最小ヒストグラム幅を付与
- `SASHIMI_FAST=1` で軽量実行（ヘッドレス、ログ出力なし）
- 図は `log/radial_pdf_evolution_fast.png` に保存

---

### 3. test_r_dependent.py（統合テスト）

**目的**: r依存計算と軌道進化PDFの動作検証、Einasto プロファイル比較。

**主要テスト**:
- `TestRDependent.test_r_dependent_production_parameters()`: 本番パラメータでのQC
  - M0 = 1e12 M☉, z=0 での subhalo_properties_r_dependence_calc() 実行
  - Einasto、NFW、計算結果を複合プロット
  - 画像は `log/combined_plot_*.png` に保存

**ユーティリティ関数**:
- `einasto(r, alpha=0.678, r_2=0.81, N=1)`: Einasto プロファイル（参照用）
- `_test_plot_hist()`: ヒストグラムプロット補助関数

---

### 4. test_sashimi.py（単体テスト）

**目的**: sashimi_c の基本機能テスト。

**主要テスト**:
- `TestSashimiC.test_mass_function_original()`: 質量関数の検証（参照データとの比較）

---

### 5. test_rpdf_integration.py（統合テスト・簡易）

**目的**: r依存計算の簡易検証（q値の有限性、正規化確認）。

**テスト**:
- `test_subhalo_rpdf_basic()`: q の有限性、範囲 [0,1] チェック

---

### 6. boost_iteration.py（パラメータスイープ）

**目的**: annihilation boost因子 B の z, ma に対する変化を計算し、データファイルに保存。

---

## 関数呼び出し関係（ワークフロー）

### ユースケース1: 標準的なサブハロー計算

```
from sashimi_c import subhalo_properties

sh = subhalo_properties()
M0 = 1e12 * sh.Msun

ma200, z_acc, rs_acc, rhos_acc, m_z0, rs_z0, rhos_z0, ct_z0, weight, survive = \
    sh.subhalo_properties_calc(M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=500)
```

**内部フロー**:
- halo_model.conc200(M, z) → 濃度
- halo_model.Mvir_from_M200() → Mvir 変換
- subhalo_mass_stripped() → tidal stripping 計算

### ユースケース2: r依存分布＋軌道進化PDF

```
sh = subhalo_properties()

ma200, z_acc, rs_acc, ..., q = \
    sh.subhalo_properties_r_dependence_calc(M0, q_bin=100, redshift=0.0, 
                                             orbit_el_mode='sample_Rc_eta')
```

**内部フロー**:
- `_subhalo_properties_r_dependence_calc()` → 軌道パラメータ計算
- `_p_q_orbit_evolved_from_racc(..., el_mode='sample_Rc_eta')` ⭐
  - `radial_pdf_evolution_in_units_of_dynamical_time_nfw_cached_phase_tables.py` をインポート
  - `sample_EL_from_eta_Rc_placed_at_Rvir()` → 初期条件生成
  - `build_orbit_cache()` → 位相テーブル
  - `evolve_r_from_cache()` → 時間進化
  - `radial_pdf()` → 分布計算

### ユースケース3: 軌道進化PDF の単独実行

```
# radial_pdf_evolution_in_units_of_dynamical_time_nfw_cached_phase_tables.py を直接実行

SASHIMI_FAST=1 python radial_pdf_evolution_in_units_of_dynamical_time_nfw_cached_phase_tables.py
```

**フロー**:
- main() → build_halo() → sample_EL_from_eta_Rc_placed_at_Rvir()
- → build_orbit_cache() → evolve_r_from_cache() → radial_pdf() → 図出力

---

## 重要なパラメータとデフォルト値

### sashimi_c.subhalo_properties.subhalo_properties_calc()
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `M0` | - | ホスト質量 (M200 @ z=0) |
| `redshift` | 0.0 | 計算対象の赤方偏移 |
| `dz` | 0.01 | z グリッド間隔 |
| `zmax` | 7.0 | 最大降着赤方偏移 |
| `N_ma` | 500 | サブハロー質量グリッド数 |
| `sigmalogc` | 0.128 | 濃度の対数正規分布σ |
| `ct_th` | 0.77 | tidal 破壊閾値 (c_t) |
| `profile_change` | True | 密度分布進化を含めるか |
| `method` | "pert2_shanks" | tidal stripping アルゴリズム |

### radial_pdf_evolution...build_orbit_cache()
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `E` | - | 軌道エネルギー |
| `L` | - | 軌道角運動量 |
| `r0` | - | 初期位置 |
| `halo` | - | NFW パラメータ辞書 |
| `n_theta` | 512 | 角度グリッド数（精度↑で遅くなる） |

### radial_pdf_evolution...radial_pdf()
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `r` | - | 半径データ |
| `nbins` | 30 | ヒストグラムビン数 |
| `rmin` | None | 最小半径（自動計算） |
| `rmax` | None | 最大半径（自動計算） |

---

## 作業ログ（2026-02-04）

### 目的
- サブハロー分布が期待される Einasto プロファイルに近づくよう、軌道初期条件とプロット系を調整。
- 解析・可視化の再現性/可読性を向上。

### 主要な変更点
- `radial_pdf_evolution_in_units_of_dynamical_time_nfw_cached_phase_tables.py`
	- 初期条件を「r=Rvir に一斉配置 + E,L を確率的にサンプル」に合わせるため、
		`sample_EL_from_eta_Rc_placed_at_Rvir(...)` を追加。
		- `Rc ~ U(0.6,1.0)Rvir`, `eta ~ Beta(2.05, C1+1)` から `E=E_circ(Rc)`, `L=eta Lc(Rc)` を作り、
			`r0=Rvir` を通過できる軌道だけ採択（`v_r^2(r0) >= 0` を満たすものをリジェクトサンプル）。
	- `main()` は上記サンプラを使用するよう変更（`sign_mode='in'` で inward）。
	- τ=0 でヒストグラムが壊れる問題に対応：`radial_pdf` で極端に狭いレンジに最小幅を付与。
	- τ=0 のデルタ状ピークにより y 軸が引きずられる問題を回避：
		τ=0 を除いた最大値で y 軸上限を設定。
	- `SASHIMI_FAST=1` で軽量/ヘッドレス実行できるようにし、図は `log/radial_pdf_evolution_fast.png` に保存。

- `sashimi_c.py`
	- r依存の軌道進化 PDF (`_p_q_orbit_evolved_from_racc`) に `el_mode` を追加。
		- **デフォルト**: `el_mode='sample_Rc_eta'`（上記の確率サンプルを使用）
		- 旧挙動: `el_mode='fixed_Ecirc_at_Rvir'`（E は固定、L のみ確率）
	- `subhalo_properties_r_dependence_calc(..., orbit_el_mode=...)` を追加し、
		上記モードを選択可能にした。

### 動作確認
- `radial_pdf_evolution_in_units_of_dynamical_time_nfw_cached_phase_tables.py` は
	`SASHIMI_FAST=1` で実行し、エラーなく完走・画像保存を確認。
- `sashimi_c._p_q_orbit_evolved_from_racc` は `el_mode` 両方でスモークテストし、
	`p` の形状と正規化（`sum(p)=1`）を確認。

### メモ
- Einasto に近づいた要因は、E と L を確率的にサンプルしつつ r=Rvir に配置する
	「通過条件付きサンプル」へ切り替えたことが大きい。
- τ=0 のヒストグラムは本質的に殻（ほぼ δ）になるため、
	描画時のスケーリング調整は必須。

---

## 作業ログ（2026-02-05）

### 目的
- r依存（orbit-evolved）計算について、サブハロー質量ビンごとの空間分布 $n(r)$ を比較できるようにする。

### 変更点
- test_r_dependent.py
  - `test_r_dependent_production_mass_bins_overlay()` を追加。
    - surviving サブハローを質量ビン（1e5–1e6, …, 1e9–1e10 Msun）で選別し、各ビンの $n(r)$（形状比較のため正規化）を重ね描き。
    - `RUN_LONG_TESTS=1` のときのみ実行される長時間テスト。
    - 図は `log/n_of_r_massbins_production_*.png` に保存。

### 動作確認
- `RUN_LONG_TESTS=1 python -m unittest -q test_r_dependent.TestRDependent.test_r_dependent_production_mass_bins_overlay` が成功し、プロットが保存されることを確認。

### 追加の動作確認
- `RUN_LONG_TESTS=1 python -m unittest -q test_r_dependent.TestRDependent.test_r_dependent_production_mass_vs_r_heatmap` が成功し、ヒートマップ `log/mass_vs_r_heatmap_production_20260205_185318.png` が保存されることを確認。