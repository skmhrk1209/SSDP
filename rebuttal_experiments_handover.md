# リバッタル用・近似誤差実験の引き継ぎ資料

論文 "Stochastic Signed Distance Processes"(arXiv 2606.20856、TMLR Paper9849)の査読対応として行っている、first-passage 確率の近似誤差の実験について、目的・設計・結果をまとめる。数値はすべて保存データから計算したもの(2026-09-23 時点)。

前提資料: 論文(https://arxiv.org/abs/2606.20856。LaTeX ソースは `curl -sL -A "Mozilla/5.0" https://arxiv.org/e-print/2606.20856 | tar -xz` の `main.tex`)、査読(https://openreview.net/forum?id=JSftvBX0lg。リポジトリ直下 `Stochastic Signed Distance Processes _ OpenReview.pdf` はその保存)。

## 1. 何のための実験か

査読の Requested Changes のうち、次の 3 つに答えるための実験である。

| 呼び名 | 査読者の要求(要旨) | 重要度 |
|---|---|---|
| rXEt-R2 | Proposition 3.1 は局所的な補間近似に依り、誤差は標本間隔が小さいと消えると論じている。導出した確率を高解像度のモンテカルロと直接比べる統制実験がほしい。標本間隔、OU の相関・ノイズのパラメータ、複数回のゼロ交差の頻度を振ると特に有用 | 指定なし |
| rXEt-R3 | negative-absorbing 近似を、その仮定が意図的に破られる場合(複数表面・薄い構造)でストレステストしてほしい | 指定なし |
| YmKF-R2 | negative-absorbing 近似の失敗例、特に複数の表面と交わるレイ・複数回の下向きゼロ交差の議論を拡充 | Critical for acceptance |

原文(回答はこの文言に対して書く):

- rXEt-R2: "Directly validating the first-passage approximations. Proposition 3.1 in the paper relies on a local interpolation approximation whose error is argued to vanish as the sampling interval becomes small. A controlled synthetic experiment comparing the derived probabilities against high-resolution Monte Carlo simulation of the stochastic process would substantially strengthen this part of the paper. It would be particularly useful to vary sample spacing, OU correlation/noise parameters, and the frequency of multiple zero crossings."
- rXEt-R3: "Stress-testing the negative-absorbing approximation in cases where its assumption is deliberately violated. The authors acknowledge that this approximation cannot model sign reversions and may be inaccurate for rays intersecting multiple object components. The current ablation shows that the approximation works well on average and reduces training time substantially, but a targeted multi-surface/thin-structure experiment would make the claimed practical trade-off much more informative."
- YmKF-R2: "Expand discussion of failure cases for the negative-absorbing approximation, especially rays with multiple surface intersections or multiple downward zero-crossings. (Critical for acceptance: the default method relies on this approximation, and its limits need clearer justification.)"

論文の該当箇所:
- 遷移核 $S_{i+1}\mid S_i=s_i\sim\mathcal N(\alpha_i s_i+\beta_i,\gamma_i)$、$\alpha_i=e^{-\kappa_i\Delta t_i}$、$\gamma_i=\tau_i^2(1-\alpha_i^2)/2\kappa_i$。事象 $\mathcal A_i$(時刻 $t_i$ まで未交差)、$\mathcal B_i^{\downarrow}=\{S_{i+1}\le0\}$、$\mathcal B_i^{\uparrow}$(区間内で入って出る)。
- Eq. (23): $P(\mathcal B_i^\downarrow\mid s_i)$(厳密)。Eq. (25) = Proposition 3.1: $P(\mathcal B_i^\uparrow\mid s_i,s_{i+1})\approx\mathbb 1(s_{i+1}>0)\exp(-2s_is_{i+1}/\Omega_i\Psi_i)$。$\Psi^{-1}\mu$ を区間内で線形補間する近似で、補間残差は $O(\Delta t_i^2)$。Eq. (26): それを $s_{i+1}$ で積分した $P(\mathcal B_i^\uparrow\mid s_i)$。
- full Bayesian formulation(3.4.2 節): survival で条件付けた分布をフィルタで伝播する。ガウスのモーメントマッチング、切断ガウス、32 点の求積という近似を含む。
- negative-absorbing 近似(3.4.3 節、論文の既定の手法): 「$S$ が一度負になったら正に戻らない」という仮定。これにより $\mathcal A_i\equiv\{S_i>0\}$、$\mathcal B_i\equiv\mathcal B_i^\downarrow$ となる。

実験では negative-absorbing 近似を 2 つの近似に分解して調べる。

- **近似 A**: survival の条件付け $\mathcal A_i$ を $\{S_i>0\}$ で置き換える(フィルタを使わない)。複数の表面があると、手前の表面を通過した履歴を無視するので破れる。
- **近似 B**: $\mathcal B_i^\uparrow$ の項(Eq. 26)を落とす。区間内で入って出る交差、すなわち薄い構造やノイズによる出入りで破れる。

| レンダラ | 近似 A | 近似 B | 論文との対応 | 図でのモデル名(✓ = その事象をモデルに入れている) | 色 |
|---|---|---|---|---|---|
| BF_UP | なし | なし | full Bayesian formulation | SSDP-Facto ($\mathcal A$ ✓, $\mathcal B^\uparrow$ ✓) | dodgerblue |
| BF | なし | あり | この実験のための構成 | SSDP-Facto ($\mathcal A$ ✓, $\mathcal B^\uparrow$ ✗) | mediumseagreen |
| NA_UP | あり | なし | この実験のための構成 | SSDP-Facto ($\mathcal A$ ✗, $\mathcal B^\uparrow$ ✓) | hotpink |
| NA | あり | あり | negative-absorbing 近似 | SSDP-Facto ($\mathcal A$ ✗, $\mathcal B^\uparrow$ ✗) | tomato |

| 実験 | 問い | 対象 |
|---|---|---|
| exp 1(interval) | Proposition 3.1 だけを単一区間で取り出したとき、Eq. (26) の誤差は $\Delta t$ とともにどう消えるか | rXEt-R2 |
| exp 2(ray, approx-A) | 近似 A の誤差は表面の数 K にどう依存するか | rXEt-R3, YmKF-R2 |
| exp 3(ray, approx-B) | 近似 B の誤差は表面の厚み W にどう依存するか | rXEt-R3 |
| ray, approx-A+B | 論文の既定の手法(A+B)そのものの誤差 | YmKF-R2 |
| 学習に沿った誤差 | 実際の学習で、場は誤差の大きい領域を通るか。収束後はどうか | 上の 3 つの補強 |
| exp 4(sdf) | 学習した場が地図の失敗領域に収束しても学習に沿った誤差が 0 なのはなぜか。近似誤差は再構成された幾何(厚さ・面の位置)にどう転化するか | rXEt-R3, YmKF-R2 |

## 2. 設計

### 2.1 共通: モンテカルロ参照と距離

**モンテカルロ(MC)参照。** SSDP のパスを各標本区間の細分割点で厳密な遷移核からサンプルし、ゼロ交差を**細分割点での符号だけ**から検出する。検証対象の式(Brownian bridge の交差確率)を参照側に使うと循環するので使わない。符号による検出は細分割の内側の交差を見落とし、m 細分割での検出確率は $P(m)=P+c_1m^{-1/2}+c_2m^{-1}+O(m^{-3/2})$ と振る舞う(細分割の中でドリフトがほぼ一定なとき)。そこで同じパスを 1, 2, 4 個おきに間引いた 3 段(n, n/2, n/4)の検出から $c_1,c_2$ を消して無限細分割へ外挿する(重みは 6.83, −8.24, 2.41)。3 段は同じパスの入れ子なので統計ノイズはほぼ打ち消し合う。

**細分割数と間引きの規則(全実験共通)。** 見落としの大きさは、細分割幅 h を相関長 1/κ で測った κh = κΔt / n で決まる(見落としの総量は 0.58 τ√h × 境界での密度、Broadie–Glasserman–Kou)。規則は 2 つ: (A) 最も細かい段で κh ≤ 1/100、(B) 外挿の最も粗い段(n/4)でも κh ≤ 1/10(展開の前提を守る)。実験中で最も大きい κΔt は exp 1 の κΔt の掃引の上端 10(学習した場は κ = 112、Δt = 1/24 で 4.7、地図は κ = 100 で 4.2)なので、(A) から n = 1000、(B) から間引きの比は 2 以下、で n = 1000・間引き (1, 2, 4) を全実験に使う(κh は最大 0.01 と 0.04)。外挿後の残差は (κh)^{3/2} のオーダーで、相対 10^{−3} 以下(MC の統計誤差より小さい)。

**距離。** レイ上の実験では、レンダラの first-passage 分布と MC 参照の分布の **Cramér 距離** $\int(F_1(t)-F_2(t))^2dt$(CDF は区間内で線形)を使う。分布は、レイ内でヒットしたという条件で正規化する。正規化は論文の CRPS の評価コード(`tools/evaluation/evaluate_uq_metrics.py`)と同じ手順で、各区間の確率を 1e-6 で下から押さえてから L1 正規化する(必ず分布になり、質量がどこにも無ければ uniform になる)。レンダラ側と MC 側の両方に適用する。
- 正規化しない分布同士の距離を使わない理由: ヒット確率の差 δ が、表面から遠端までの長さを掛けた δ² として入り、値がレイの遠端の位置に依存する。
- 正規化した距離はヒット確率の誤差を含まない。そこで、レイ内で first passage が起きる確率(ヒット確率)の**二乗距離** $(p_F - p_Q)^2$ を併せて使う($p_F$ はレンダラ、$p_Q$ は MC のヒット確率)。Brier スコアに付随する発散で、Cramér 距離が CRPS に付随する発散であるのと同じ関係にある。ヒット確率そのものも保存してある(`hit_prob_1`、`hit_prob_2`)。

**パス数の決め方と MC の標準誤差。** パス数は、上の細分割数のもとで計算時間から決めた(1 ジョブが 1 時間以内、10 の冪)。基準にする誤差は MC の統計誤差(標準誤差)で、図に描く測定誤差はこれだけである。細分割の残差と式側の求積誤差は描かないので、標準誤差の 1/3 以下にして帯に隠れるようにする。MC が直接推定するのは、exp 1 では質量 $p$、レイ上の実験では first-passage の CDF $F_k$ で、どちらもパスの割合なので標準誤差は $\sqrt{q(1-q)/N}$、上界は $1/(2\sqrt N)$(外挿前。外挿後は 3 つの検出の線形結合になるので、これをわずかに上回る)。決めたパス数での標準誤差は次のとおり。

| 実験 | パス数 $N$ | 上流(確率)の標準誤差 | 下流(最終指標)の標準誤差 |
|---|---|---|---|
| exp 1 | $10^8$ / 組み合わせ | $\sqrt{p/N}$: $p=10^{-3}$ で $3\times10^{-6}$、$10^{-4}$ で $10^{-6}$ | 質量の差にそのまま乗る。図に `MCSE ($\times 2$)` の帯として描く(外挿後の厳密な分散) |
| 地図 | $10^5$ / 格子点 | CDF: $\le 1.6\times10^{-3}$ | 格子点・レンダラごとに計算して記録(下の段落。値は 3.2 節) |
| 学習に沿った誤差 | $10^3$ / レイ | CDF: $\le 1.6\times10^{-2}$ | レイごとに計算し、距離と同じ分位で記録(下の段落。値は 3.3 節) |

**下流の標準誤差の計算(地図・学習)。** 外挿した参照の CDF は、パスごとの指標(3 段の検出の指標を外挿の重みで結合したもの)の平均なので、その共分散 $\Sigma$(標本点 × 標本点)はパスごとの指標の標本共分散をパス数で割ったもので厳密に出る(exp 1 の質量の分散と同じ扱い)。距離はこの CDF の非線形関数(正規化と 2 次形式)なので、参照のノイズをガウスとして 2 次まで展開し、勾配 $g$ とヘッセ行列 $H$(自動微分)から、バイアス $\mathrm{tr}(H\Sigma)/2$(レンダラが参照と一致する点に残る床)と分散 $g^\top\Sigma g+\mathrm{tr}((H\Sigma)^2)/2$ を各格子点・各レイで計算し、`<指標>_stderr`、`<指標>_bias` として記録する(`_compute_metric_errors`)。ヒット確率の標準誤差 `hit_prob_2_stderr` も記録する。検算(N = 2000、細分割 100 の小設定、4 つの格子点、NA と BF_UP): 24 シードのばらつきに対して予測した標準誤差は CDF・距離とも 15% 以内(距離が大きい角 κ = 100、τ² = 10 の NA では 35% 過大)、床は 24 シードの平均と 2×10⁵ パスの値の差(分解能 2×10⁻⁴)と矛盾しない。exp 1 の厳密な標準誤差も 8 シードのばらつきと一致する。図には描かず、値は 3.2 節と 3.3 節に書く。

図に MC の誤差を描くのは exp 1 だけ(解像した母集団)。

### 2.2 exp 1: 単一区間での Proposition 3.1

区間 $[0,\Delta t]$ で $\kappa,\tau$ は定数、平均は $\mu(t)=\mu_0+at+bt^2$、$S_0$ は $\mathcal N(\mu_0,\sigma^2)$ を $S_0>0$ に切断したもの(区間の始点で path が生きている。論文の Eq. (35) が $s_i>0$ で評価するのと同じ条件付け)とする。比べるのは up-crossing の確率 $P(\mathcal B^\uparrow\mid S_0>0)$ で、フィルタも他の区間もシーンも関与しない。

- 真値: $P(\inf_{(0,\Delta t]}S\le0,\ S(\Delta t)>0\mid S_0>0)$ を MC で推定($S_0$ は切断正規分布から逆関数法で引く)
- 式: Eq. (26) を切断正規分布で平均したもの(`SSDP._get_log_up_cross_prob` をそのまま呼ぶ)。積分は ssdp.py の full Bayesian formulation と同じ方式で、$S_0>0$ に切断した正規分布の CDF 空間で Gauss–Legendre、節点 1000。節点数の規則は「中点則 $2^{18}$ 点(S₀ の ±10σ。$2^{20}$ 点・±14σ との差は $10^{-10}$ 以下)との差が、全組み合わせ(確率 ≥ 1/N)・$\widetilde{\Omega}\in\{0.1,1,10\}$(掃引の両端と中央。$\kappa\Delta t$ の掃引の両端と中央 $\widetilde{\Omega}\in\{0.22,6.4,4.9\times10^8\}$ でも確認)で MC の標準誤差の 1/3 以下になる最小の 10 の冪」で、`check_quadrature.py` の結果(`data/checks/quadrature.json`)は 10 節点で最悪 470 標準誤差、100 節点で最悪 2.2($\widetilde{\Omega}=10$)、1000 節点で最悪 0.0015(中央値 $4\times10^{-5}$ 以下)。収束は節点数の 2 乗に反比例する代数的なもので、論文の 32 節点では最悪 16 標準誤差。

**還元補題**(証明の要点: 時間を相関長 $1/\kappa$ で、振幅を $\sigma_{st}$ で測り直すと、残差は $\kappa=1$・定常分散 1 の標準 OU になり、平均は $\tilde\mu_0+\tilde a s+\tilde b s^2$、始点は $\mathcal N(\tilde\mu_0,\tilde\sigma^2)$ になる。この変換は時間の単調な付け替えと正の定数での除算なので、パスの符号・下限・端点の符号、したがって事象 $\mathcal B^\uparrow$ と $\{S_0>0\}$ を保ち、式側も同じ変数の関数になる): 両者は 5 つの無次元量 $x=\kappa\Delta t$(以下では同じ情報を持つ $\Omega/\sigma_{st}^2=e^{2\kappa\Delta t}-1$ で表す)、$\tilde\mu_0=\mu_0/\sigma_{st}$、$\tilde\sigma=\sigma/\sigma_{st}$、$\tilde a=a/(\kappa\sigma_{st})$、$\tilde b=b/(\kappa^2\sigma_{st})$($\sigma_{st}=\tau/\sqrt{2\kappa}$)だけの関数である。したがって $(\kappa,\tau)$ を独立に振る必要はなく、標準形($\kappa=1,\tau^2=2$)で無次元空間を掃けばよい。査読者の言う「標本間隔」は $x$、「OU の相関・ノイズ」は $x$ と $\tilde\mu_0,\tilde\sigma$、「平均の形」は $\tilde a,\tilde b$ に対応する。

**掃引の変数**: 区間の長さは $\Omega_i=\Theta(t_{i+1})$(Proposition 3.1 の証明で、残差のマルチンゲール部分の二次変分 = Brownian bridge の背後のブラウン運動の分散。$\kappa,\tau$ 一定なら $\Omega=\sigma_{st}^2(e^{2\kappa\Delta t}-1)$)を $\sigma_{st}^2$ で割った $\widetilde{\Omega}$ で振る。補間残差の上界($\Omega^2\sup|h''|/8$)も bridge の交差確率($e^{-2ab'/\Omega}$)も $\Omega$ で厳密なので、理論の変数はこれである($\kappa\Delta t\ll1$ では $\widetilde{\Omega}\approx2\kappa\Delta t$)。

**掃引**: $\widetilde{\Omega}$ は $10^{-1}$ から $10^1$ まで対数等間隔の 11 点(1 桁に 5 点。$\kappa\Delta t$ では 0.048〜1.2)。1 を中心に 1 桁ずつ。下端は MC の分解能で決まる(誤差は $\widetilde{\Omega}^2$ で減り、$\widetilde{\Omega}=0.1$ で中央値が $10^8$ パスの標準誤差の 2 倍と同程度になる。それより下は $10^{10}$ パスが要り予算外)。上端は、残差の展開が効く $\widetilde{\Omega}<1$ の 1 桁上で、誤差の増加が止まる($\widetilde{\Omega}\gtrsim10$ で頭打ち。$10^3$ まで振った確認では中央値 2〜3% で飽和)。**normalized sampling interval $\widetilde{\Delta t}=\kappa\Delta t$ の軸の図のために、$\widetilde{\Delta t}$ を $10^{-1}$ から $10^1$ まで対数等間隔に振った掃引(11 点)を別に持つ**(評価器の入力は $\widetilde{\Delta t}$ で、$\widetilde{\Omega}$ の掃引は起動スクリプトが $\widetilde{\Delta t}=\log(1+\widetilde{\Omega})/2$ に換算して渡す。記録には両方の値が入る。$\widetilde{\Delta t}>1.2$ の点は $\widetilde{\Omega}>10$ で、誤差が頭打ちの先にある領域)。$\tilde\mu_0,\tilde\sigma\in\{0.1,1,10\}$、$\tilde a\in\{0,-0.1,-1,-10\}$、$\tilde b\in\{0,0.1,1,10\}$(10 の冪。効くのは区間内の平均の動きとノイズの比なので対数で振る)。全 144 通り。定常標準偏差のまわり 3 桁で、学習した場の値(Δt = 1/24、SDF の傾き ±1 として、$|\tilde a|$ は学習初期 0.8〜1.2、収束後 10〜30。$\tilde\sigma$ は $\sigma_0=0.1$ 固定なので初期 0.03〜収束後 60)を覆う。平均の形は「生きている path が表面に近づく区間」のもの: 始点で境界の上($\tilde\mu_0>0$)、上がらない($\tilde a\le0$。$\tilde a=0$ は近づかない対照)、上に曲がるか曲がらない($\tilde b\ge0$。下に曲がる平均は戻らないので、up-crossing はノイズによるものだけになり、$\tilde b=0$ で代表できる)。float64、1000 細分割・間引き (1, 2, 4)(2.1 節の規則)。$R(t)=e^{-\kappa t}R(0)+Z(t)$ と分解し、同じ $Z$ と同じ一様乱数(切断正規分布の逆関数に通す)を 144 通りすべてで共有する。パス数は全点で $10^8$、シードは 1 つ(42)。

**補間残差の主項**(Proposition 3.1 の証明から): 線形補間されるのは $h(\omega)=\Psi(t_i,t)^{-1}\mu(t)$ を二次変分の時計 $\omega=\Theta(t)$ で見た関数。無次元化して展開すると $\tilde h(\tilde\omega)=\tilde\mu_0+(\tilde\mu_0+\tilde a)\tilde\omega/2+(\tilde b-\tilde\mu_0/2)\tilde\omega^2/4+O(\tilde\omega^3)$ で、残差の主項の係数は $\tilde b-\tilde\mu_0/2$(平均の曲率から OU の引き戻し分を引いたもの。傾き $\tilde a$ は主項に入らない)。したがって平均が線形でも残差は $\tilde\mu_0/2$ に比例して残り、$\tilde b>\tilde\mu_0/2$ で式は過小、$\tilde b<\tilde\mu_0/2$ で過大になる。大きさは $|\tilde\delta|\le\widetilde{\Omega}^2|\tilde b-\tilde\mu_0/2|/16$。

**確率の誤差の次数**(残差からの見積もり。論文が述べているのは残差の次数まで): 交差できるのは始点 $S_0$ が境界からノイズの幅 $\sqrt\Omega$ 以内にある経路だけで、残差はそれを $\Omega^2$ だけ、すなわちその幅の $\Omega^{1.5}$ 倍だけずらす。したがって**相対誤差は $O(\Omega^{1.5})$**。$S_0$ の密度が幅 $\sqrt\Omega$ の上で平らなら(条件: 密度の相対変化 $\widetilde{\Omega}/2\tilde\sigma^2+\tilde\mu_0\sqrt{\widetilde{\Omega}}/\tilde\sigma^2\ll1$)、確率そのものが $O(\Omega^{0.5})$ なので**絶対誤差は $O(\Omega^2)$**(残差と同じ次数)。これはすべての組み合わせの $\Omega\to0$ での極限だが、掃引の範囲では $\tilde\sigma=10$ の組み合わせだけがこの領域にあり、始点分布が幅 $\sqrt\Omega$ の内側に収まる組み合わせ($\tilde\sigma,\tilde\mu_0\lesssim\sqrt{\widetilde{\Omega}}$)では確率が O(1) で絶対誤差も $O(\Omega^{1.5})$、境界が遠く密度の裾が届かない組み合わせでは $e^{-2\tilde\mu_0^2/\widetilde{\Omega}}$ で急減する。float64、1000 細分割・間引き (1, 2, 4)(2.1 節の規則)。$R(t)=e^{-\kappa t}R(0)+Z(t)$ と分解し、同じ $Z$ と同じ一様乱数(切断正規分布の逆関数に通す)を 144 通りすべてで共有する。パス数は全 $x$ で $10^8$、シードは 1 つ(42)。

**検算**: (i) 同じ無次元群に落ちる 4 つの物理設定 $(\kappa,\tau^2)=(1,2),(10,0.25),(100,0.02),(0.1,8)$ で、MC の推定値と式の値が標準形と一致することを確かめる(`check_invariance.py`)。(ii) 下向き交差の質量は細分割が不要で Eq. (23) が厳密なので、式と MC の差を標準誤差で割った z が標準正規に従うことでパスを検算する。結果は 3.1 節。

**図**: 横軸は normalized quadratic variation $\widetilde{\Omega}$ と normalized sampling interval $\widetilde{\Delta t}$ の 2 通りで、それぞれ自分の掃引(11 点)から描く。縦軸は絶対誤差 |式 − MC|(論文に載せるのはこちら。相対誤差の図も出力する)で、レンジは指標ごとに 2 軸で共通(5% 点の最小から 95% 点の最大を含む 10 の冪)。描くのは、全 11 点で参照の確率が正かつ相対標準誤差 ≤ 1% の組み合わせ(参照の精度だけによる選別で、誤差の大きさでは選別しない)。組み合わせごとの線と、その中央値・25〜75%・5〜95%、MC の標準誤差の 2 倍、横軸の 2 乗(絶対)・1.5 乗(相対)に比例するガイド(上の次数)を描く。

### 2.3 exp 2 / exp 3 / A+B: 解析的なスラブ上の誤差地図

- **場**: x 軸に垂直な K 枚のスラブの解析的な SDF を平均 $\mu$ とし、$\kappa,\tau$ を全区間で一定にした SSDP(`AnalyticSSDP`。`SSDP` を継承し、レンダラは `ssdp` の実装をそのまま使う)。学習を介さないので、レンダラの近似誤差だけが出る。
- **レイ**: スラブを垂直に貫く 1 本。$x\in[-1,1]$、N = 48 標本(論文の設定)、$\Delta=2/48$、$\sigma_0^2=0.01$(学習の設定と同じ)。遷移分散の下限は $10^{-12}$: 論文の設定の $10^{-6}$ では格子の τ が小さい側(τ ≲ 0.01。γ は τ = 0.001 で $5\times10^{-9}$ まで下がる)で下限が効いてしまうので、格子全体で効かない値にする。
- **シーン**: K ∈ {1, 3}(中心は 0、または −0.5, 0, 0.5)× W ∈ {0.05 = 1.2Δ, 0.025 = 0.6Δ}。K は近似 A(複数表面)、W は近似 B(薄い構造)を破るための軸。
- **位相**: スラブの中心と標本点の位置関係(評価器の `phase`。標本点を標本区間の phase 倍だけずらす)。phase 0.0 はスラブの中心が標本点上にあり、スラブの中に必ず標本点が入る。phase 0.5 はスラブの中心が標本区間の中央にあり、標本点から最も見えにくい配置で、W = 0.6Δ ではスラブが 2 つの標本点の間に完全に入る。両方を評価し、図も両方を作る。
- **格子**: $(\kappa,\tau)$ の平面。κ ∈ [1e-2, 1e2] × τ ∈ [1e-3, 1e1]、どちらも log 等間隔で 11 点(11 × 11 = 121 点)。学習で場が訪れる範囲(κ 0.027〜112、τ 0.0058〜2.6)を覆う(κ の上端を超える分は地図の端に張り付けて描く)。
- **MC**: 1 格子点あたり $10^5$ パス、1 区間 1000 細分割・間引き (1, 2, 4)(2.1 節の規則)。1 格子点につき MC 参照は 1 回だけ計算し、4 つのレンダラすべてを同じ参照と比べる(乱数シードも格子点ごとに同じ値に戻す)。
- **図**(1 位相・1 シーン・1 対・1 指標につき 3 枚): 近似なしのレンダラの MC との距離、近似ありのレンダラの MC との距離、その差(あり − なし)。カラースケールは指標ごとに全図で共通で、距離は 0〜M、差はその折り返し ±M。M は「各地図の 75% 点の最大(全位相・全シーン・全レンダラ。起動スクリプトの `MAP_QUANTILE`)」を、距離と差分の両方のカラーバーの目盛(matplotlib の自動選択)が M で終わるまで広げた値。対は、A: (BF_UP, NA_UP) と (BF, NA)、B: (BF_UP, BF) と (NA_UP, NA)、A+B: (BF_UP, NA)。距離の地図にはそのレンダラで学習した場のパラメータの軌跡(2.4 節。全チェックポイント)を、差の地図には近似ありのレンダラで学習した場の軌跡を重ねる。

### 2.4 学習に沿った誤差

- **ラン**: 4 シーン × 4 レンダラで学習した場(`outputs/TMLR-toy-analysis/ssdp-facto-half/ssdp-facto-half-<scene>-{NA,NA-UP,BF,BF-UP}/202604/nerfstudio/`)。10000 反復、48 標本/レイ、100 ステップごとのチェックポイント 100 個、学習シード 1 つ。学習スクリプトは `tools/training/scripts/toy/train_ssdp_facto_{na,na_up,bf,bf_up}.sh`、データは `datasets/nerfstudio/sdfstudio/toy/cuboid/<scene>/`。
- **学習されたパラメータ**(地図に重ねる軌跡): x 軸に平行な 81 本のレイで、各レイの区間にわたる中央値をレイで平均した $(\alpha,\gamma,\kappa,\tau^2)$。
- **誤差**: 評価カメラ 64 台の画像(1000 × 1000)を 100 画素おきに間引いた格子(10 × 10 × 64 = 6400 本)のうち、GT メッシュに当たるレイ(論文の不確実性評価と同じ基準。決定的で、乱数による選択はしない。K = 1 で 960 / 946 本、K = 3 で 1930 / 1860 本、W = 0.05 / 0.025 の順)。各チェックポイントで、**その時点の学習に使っているレンダラ**(遷移分散の下限も学習時の設定 $10^{-6}$ のまま)と、**同じ学習済みの場の MC**($10^3$ パス/レイ、1000 細分割・間引き (1, 2, 4)、外挿あり)の距離を測り、レイにわたる中央値と 25〜75% を描く。レイをヒット確率で選別しない。1 ジョブは 10 チェックポイントで、1 ランの 10 個の出力を `merge_training_trajectory.sh` で結合する。
- 図: 横軸 学習ステップ、縦軸は誤差のレイにわたる中央値と 25〜75%(レンジは指標ごとに全図で共通、0 から全ランの 75% 点の最大を含む目盛まで)。全レンダラの中央値が落ち着いた最初のチェックポイントの次までを拡大した窓つき(縦軸は窓内の 75% 点の最大を含む目盛まで)。目盛は matplotlib の自動選択。

### 2.5 exp 4: 再構成された幾何

- **問い**: W = 0.6Δ のランは地図の「どのレンダラも捉えない領域」(κ 大、τ 小、phase 0.5)に収束するのに、学習に沿った誤差は 0 である(3.2 節)。レンダラが薄い構造を描けないなら、学習は幾何の側を変えているはずで、その大きさを測る。
- **対象**: K = 1 の 2 シーン(W = 0.05、0.025)× 4 レンダラの最終チェックポイント(step 9999)の学習した平均 SDF。MC は使わない(ネットワークの決定的な評価)。
- **厚さ**: x 軸に平行なレイを $(y,z)\in$ linspace(−0.4, 0.4, 81)² の格子(6561 本。辺から 0.1 内側)に置き、スラブ中心 ±0.05 を 101 点で標本化した学習 SDF が負になる区間の長さ(零点は線形補間)。負の区間が無いレイは「消失」として数える。真の厚さは 0.05 / 0.025。
- **面の誤差**: 同じ格子を真の前面・後面(x = ±半幅)に置いた学習 SDF。真の SDF はそこで 0 なので絶対値が面の誤差、符号が側(負 = 学習した面が外側)。
- **断面**: z = 0 の平面上、x ∈ [−0.05, 0.05]、y ∈ [−0.5, 0.5] を各 1001 点で標本化した学習 SDF と、その零等値線(学習: 実線、真値: 破線)。色は学習 SDF の負(レンダラの色)・正(灰)で、レンジは 8 断面をまとめた 5% 点と 95% 点の大きい方を 0 対称に。
- **座標系**: 位置はすべてデータセットの世界座標(真値メッシュの座標)で定め、dataparser がポーズに掛けた変換で学習の座標に写す(この toy では単位行列。学習軌跡の評価器も同じ扱い)。

## 3. 結果

数値は `data/approx-errors/summary.log`(`summarize_approx_error.sh`。最終設定のデータから計算)。図は `outputs/TMLR-toy-analysis/figures/{interval,ray}/`。

### 3.1 exp 1

図: `interval/axis-{normalized_quadratic_variation,normalized_sampling_interval}/analytic_up_cross_prob_{abs,rel}_error_plot_vs_MC.pdf`。数値は `summary.log` の exp 1 の節(掃引ごとに 1 ブロック。傾きはその軸の値が 1 未満の 5 点の中央値で当てはめる)。描かれる母集団は、全 11 点で参照の確率が正かつ相対標準誤差 ≤ 1% の組み合わせ。$\widetilde{\Omega}$ の掃引で外れる 50 通りは、境界が遠く区間内で交差しない $\tilde\mu_0=10$ かつ $\tilde\sigma\le1$(32 通り)と、平均が下がって戻らず長い区間で up-crossing の質量が消える $\tilde a=-10$ かつ $\tilde b\le1$($\tilde\mu_0\le1$ の 18 通り)。

**$\widetilde{\Omega}$ の掃引(144 通り中 94 通り)**

| $\widetilde{\Omega}$ | 0.1 | 0.25 | 0.63 | 1 | 2.5 | 10 |
|---|---|---|---|---|---|---|
| 絶対誤差の中央値 | 3.6e-5 | 1.4e-4 | 1.3e-3 | 2.6e-3 | 6.8e-3 | 1.6e-2 |
| 75% | 1.7e-4 | 1.1e-3 | 4.9e-3 | 7.6e-3 | 1.9e-2 | 5.2e-2 |
| 95% | 2.0e-3 | 1.2e-2 | 5.4e-2 | 9.7e-2 | 1.7e-1 | 4.3e-1 |
| 相対誤差の中央値 | 0.3% | 0.7% | 2.3% | 3.8% | 7.9% | 15% |
| MCSE × 2(中央値) | 2.7e-5 | 4.1e-5 | 6.3e-5 | 6.3e-5 | 6.9e-5 | 8.5e-5 |

**$\widetilde{\Delta t}$ の掃引(144 通り中 85 通り。$\widetilde{\Delta t}>1.2$ は $\widetilde{\Omega}>10$ なので、そこで質量が消える組み合わせがさらに 9 通り外れる。$\tilde a=-1$ かつ $\tilde b=0$ の 6 通りなど)**

| $\widetilde{\Delta t}$ | 0.1 | 0.25 | 0.63 | 1 | 2.5 | 10 |
|---|---|---|---|---|---|---|
| 絶対誤差の中央値 | 1.2e-4 | 1.5e-3 | 8.2e-3 | 1.8e-2 | 2.7e-2 | 3.8e-1 |
| 75% | 9.1e-4 | 5.3e-3 | 2.1e-2 | 4.5e-2 | 1.0e-1 | 7.1e-1 |
| 95% | 9.3e-3 | 5.7e-2 | 1.8e-1 | 3.7e-1 | 4.8e-1 | 9.6e-1 |
| 相対誤差の中央値 | 0.6% | 2.5% | 7.9% | 13% | 24% | 71% |

- **短い区間では Eq. (26) の誤差は区間の長さの 2 乗で消える。** 中央値の傾き(log-log、軸の値 < 1)は $\widetilde{\Omega}$ の掃引で 1.97、$\widetilde{\Delta t}$ の掃引で 2.34。$\widetilde{\Omega}$ の軸では中央値がガイド $\widetilde{\Omega}^2$ に $\widetilde{\Omega}\approx0.6$ まで乗って曲がり始め、$\widetilde{\Delta t}$ の軸では $(\widetilde{\Delta t})^2$ に $\widetilde{\Delta t}\approx1$ まで乗る($\widetilde{\Omega}=e^{2\widetilde{\Delta t}}-1$ が $\widetilde{\Delta t}\gtrsim0.3$ で $2\widetilde{\Delta t}$ より急に伸びるため)。左端では母集団の 4 割($\widetilde{\Omega}$)、2 割($\widetilde{\Delta t}$)が MC の分解能(MCSE × 2 ≈ 3e-5)の中にあり、それより短い区間は $10^8$ パスでは分解できない(2.2 節の下端の根拠)。
- **長い区間では消えない。** $\widetilde{\Omega}$ の掃引では中央値が $\widetilde{\Omega}\approx10$ で 1.6e-2 に達して増加が止まる(掃引内の最大 0.866)。$\widetilde{\Delta t}$ の掃引は $\widetilde{\Omega}$ で $5\times10^8$ まで届き、中央値は $\widetilde{\Delta t}=10$ で 0.38、95% 点は 0.96、最大は 0.996($\widetilde{\Delta t}=2.5$、$\tilde\mu_0=1$、$\tilde\sigma=0.1$、$\tilde a=-10$、$\tilde b=10$)。この組み合わせは平均が区間の途中で $-1.5\sigma_{st}$ まで下がって戻り、$\widetilde{\Delta t}\ge1.6$ では両端が正なので線形補間は交差を見ず、真の確率 0.998 に対して式は 0.002〜0.92 になる(薄い構造の失敗。地図の近似 B と同じ機構)。
- **誤差の符号と係数は $\tilde b-\tilde\mu_0/2$ で決まる(2.2 節の主項)。** $\widetilde{\Omega}=0.1$ での符号付き相対誤差(式 − MC、中央値)は、$\tilde\mu_0=0.1$ で $\tilde b=0$ なら +5e-4、$\tilde b=10$ なら −2.2%(過小)、$\tilde\mu_0=10$ では $\tilde b\le1$ で +1.1〜1.3%(過大)、$\tilde b=10$ で −1.0%。$\tilde\mu_0=1$ では $\tilde b\le0.1$ で正、$\tilde b\ge1$ で負($\tilde b=10$ で −2.3%)。
- **次数の検証**(2.2 節の見積もりの確認。`summary.log` の checks の節の $(\tilde\sigma,\tilde\mu_0)$ ごとのブロック。母集団の全 $\tilde b$ をまとめ、$\widetilde{\Omega}<1$ の 5 点の中央値で当てはめた傾き): 密度が平らな $\tilde\sigma=10$ で $\tilde\mu_0=0.1$ は 2.05(予測 2)、$\tilde\mu_0=1$ は 1.41、$\tilde\mu_0=10$ は 1.25(補間残差の高次項の係数が $\tilde\mu_0$ に比例して大きく、掃引の範囲では漸近の前段階)。$\tilde\sigma=1$ は 1.89($\tilde\mu_0=0.1$)、1.67($\tilde\mu_0=1$)。境界が遠い $\tilde\sigma=0.1$、$\tilde\mu_0=1$ は 2.96(予測: 2 より急)。$\tilde\sigma=\tilde\mu_0=0.1$ は −0.27 で、この領域の誤差は MC の分解能の中にあり傾きは測れない。
- **検算**: (i) 4 つの物理設定で MC の推定値はビット単位で一致し、式の差は最大 2e-16。(ii) 下向き交差の z は両掃引 1913 個で標準偏差 1.03、|z| < 2 が 95.4%。

### 3.2 誤差地図

以下「差」は(近似あり − 近似なし)で、正ならその近似が悪くしている。格子は 121 点。

**近似 A(survival の条件付けを落とす。approx-A、phase 0.0)**
- **K = 1 では効かない。** NA − BF の Cramér 距離の差は最大 0.010、95% 点 0.008。二乗距離では逆に BF がわずかに悪い(中央値 −0.003〜−0.004、最小 −0.031): フィルタのガウス近似の分で、近似 A の効果ではない。
- **K = 3 では、左下(κ ≤ 1、τ ≤ 0.05。α ≈ 1、γ 小の強相関)で NA が破れる。** Cramér 距離の差は最大 0.053(W = 0.05)/ 0.077(W = 0.025)、95% 点 0.051 / 0.074、格子の中央値でも 0.003。二乗距離の差は最大 0.115 / 0.119 で、この領域の NA のヒット確率は 0.94 に対し MC は 0.60。機構: 1 枚目のスラブを抜けた(すでに負になった)パスを条件付けずに 2 枚目・3 枚目でもう一度「交差」として数える二重計上で、正しい first passage の分布より手前に質量が寄り、ヒット確率が過大になる。NA_UP − BF_UP でも同じ(最大 0.054 / 0.078)。格子全体で NA のヒット確率の最小は 0.89〜0.91、MC は 0.55〜0.60。
- phase 0.5 でも同じ領域・同じ大きさ(K = 3 の最大 0.076 / 0.097)。格子全体の Cramér 距離の中央値は、K = 3 phase 0.0 で NA 0.010〜0.015、BF 0.002、NA_UP 0.003、BF_UP 0.0001。

**近似 B($\mathcal B^\uparrow$ を落とす。approx-B、phase 0.5)**
- **W = 1.2Δ(0.05)では小さい。** BF − BF_UP の Cramér 距離の差は最大 0.104(K = 1)/ 0.083(K = 3)だが 95% 点は 0.025 / 0.022 で、大きいのは κΔt > 1 の上端付近だけ。二乗距離の差は最大 0.044 / 0.013。
- **W = 0.6Δ(0.025)では、スラブが 2 つの標本点の間に完全に入り、$\mathcal B^\uparrow$ の有無が効く。** 上側の帯(κ ≳ 3、τ 0.02〜1)で BF_UP は $\mathcal B^\uparrow$ によってスラブを捉え、BF は捉えない: 二乗距離の差は最大 0.971(K = 1)/ 0.982(K = 3)(BF のヒット確率 ≈ 0 に対し BF_UP ≈ 1)、95% 点 0.21 / 0.075。Cramér 距離の差は最大 0.160 / 0.249。NA − NA_UP も同じ(二乗距離の差の最大 0.975 / 0.982)。
- **どのレンダラも捉えない領域(左上: κ ≳ 2、τ ≲ 0.02)。** ノイズが小さいと過程はほぼ平均どおりに動き、平均(SDF)の凹みが標本点の間にあるので、標本点での符号も Proposition 3.1(端点の値から補間)も交差を見ない。4 つのレンダラすべてでヒット確率が 0(MC は 1)、二乗距離 1.0、条件付き分布は一様(1e-6 の下限の正規化)対ステップで Cramér 距離 0.295。これは近似 A・B ではなく標本化の限界で、差の地図ではキャンセルして見えない。
- phase 0.0(スラブの中心が標本点上)では W = 0.6Δ でも捉える(全レンダラのヒット確率の最小 ≥ 0.41)ので、近似 B の効果は小さい(Cramér 距離の差の 95% 点 ≤ 0.025)。

**A + B(論文の既定の NA、approx-A+B/BF-UP_vs_NA)**
- NA − BF_UP の地図は、上の 2 つの重ね合わせ: K = 3 の左下(近似 A)と、W = 0.6Δ の phase 0.5 の上側の帯(近似 B)。Cramér 距離の差の最大は phase 0.0 で 0.107(K = 1)、0.088 / 0.091(K = 3)、phase 0.5 で 0.104 / 0.160(K = 1)、0.083 / 0.249(K = 3)。二乗距離の差の最大は phase 0.0 で 0.020〜0.118、phase 0.5 で 0.044 / 0.971 / 0.065 / 0.982(K1W0.05 / K1W0.025 / K3W0.05 / K3W0.025)。

**κΔt > 1 の上端(全レンダラ、全位相)**
- 地図の上 2 行(κ = 39.8、100。κΔt = 1.7、4.2)の τ ≳ 1 では、4 つのレンダラすべてが MC から離れ、Cramér 距離の最大は K = 1 で NA・BF 0.20(κ = 100、τ = 3.98)、NA_UP・BF_UP 0.46(κ = 100、τ = 1.58)、K = 3 で 0.09 / 0.16。$\mathcal B^\uparrow$ を持つ方が悪い(差の地図の右上の角で BF − BF_UP が −0.39〜−0.40)。Proposition 3.1 は区間が相関長より短いことを前提にしており(exp 1 でも κΔt > 1 で誤差が消えない)、その外では $\mathcal B^\uparrow$ の項が交差を過大に入れる。学習した場はこの角(κ ≥ 40 かつ τ ≥ 1)には入らない(3.3 節)。

**MC の誤差(2.1 節の伝播、`summary.log`)**: 距離の標準誤差は格子とレンダラにわたる中央値で Cramér $1\times10^{-5}$〜$1\times10^{-4}$、二乗 $3\times10^{-5}$〜$3\times10^{-4}$、最大は Cramér $9\times10^{-4}$、二乗 $1.4\times10^{-3}$(距離の大きい失敗点)。床(バイアス)は最大 $3\times10^{-6}$。上の差の中央値($10^{-3}$ 台)は標準誤差の 10 倍以上ある。

**軌跡との関係**: どのランも右下(κ 小、τ 大)から左上(κ 大、τ 小)へ動き、終点は W = 0.6Δ の phase 0.5 では「どのレンダラも捉えない領域」の中か境界にある。一方、学習に沿った誤差(3.3 節)は step 400〜600 以降 0 である。カメラのレイは位相も角度もまちまちで、学習した場も解析的なスラブではないので、地図の 1 本のレイの最悪の配置(phase 0.5)がそのまま現れるわけではない。学習が幾何を厚くして自分のレンダラが見える形にしていることを exp 4(3.4 節)が示す。

### 3.3 学習に沿った誤差

- **学習に使ったレンダラ自身の誤差は、最初の数百ステップだけ大きく、その後は 0 に落ちる。** 4 シーン × 4 レンダラのすべてで、中央値が $10^{-3}$ 未満に落ち着くのは step 300〜600(BF_UP は最初から $10^{-3}$ 程度)。最後のチェックポイント(step 9999)では中央値・75% 点とも $10^{-5}$ 以下。学習後の場は τ が 0.006〜0.012、κ が 40〜110 と、ほぼ決定的な過程になっており、近似 A・B のどちらも効かない。
- **Cramér 距離(中央値のピーク)**: BF_UP 0.001(全シーン)、NA_UP 0.003〜0.004(K = 1)/ 0.012(K = 3)、NA 0.011(K = 1)/ 0.015〜0.016(K = 3)、BF 0.021〜0.022(K = 1)/ 0.042〜0.046(K = 3)。近似 B だけを持つ BF が最も大きく、K = 3 で倍になる。75% 点の最大は BF で 0.031〜0.032(K = 1)/ 0.069〜0.074(K = 3)。
- **二乗距離(中央値のピーク)**: NA_UP が最大で 0.126〜0.128(K = 1)/ 0.138〜0.139(K = 3)、NA 0.027 / 0.035、BF 0.022〜0.030 / 0.021〜0.024、BF_UP 0.003 / 0.008〜0.009。NA_UP は survival の条件付けなしに $\mathcal B^\uparrow$ を足すので同じ交差を二重に数え、ヒット確率を過大評価する(地図でも同じ。3.2 節)。Cramér 距離は条件付けで正規化するのでこれを隠す。75% 点の最大は NA_UP で 0.145 / 0.160。
- **学習されたパラメータ**(地図に重ねる軌跡): 全チェックポイントで κ 0.027〜112、τ 0.0058〜2.6。どのランも、step 100 の(κ 小、τ 大 = 地図の右下)から step 500〜1000 までに(κ 大、τ 小 = 左上)へ動き、その後は左上に留まる。κ が地図の上端 100 を超えるのは k1_w0.05 の NA_UP(最大 112、step 7300)と k1_w0.025 の NA_UP(最大 106、step 5900)。
- **MC の誤差(2.1 節の伝播、`summary.log`)**: レイ 1 本あたりの距離の標準誤差は、中央値のピークの step で Cramér $7\times10^{-4}$〜$6\times10^{-3}$、二乗 $1\times10^{-3}$〜$1.2\times10^{-2}$(レイにわたる中央値。距離の中央値の 1/3〜1/10。BF_UP では距離と同程度)、床は $10^{-4}$ 台。図に出すのはレイにわたる中央値と四分位で、レイごとの MC のノイズは独立なので、それらの誤差はさらに $1/\sqrt{\text{レイ数}}$(946〜1930 本)程度に落ちる。BF_UP の $10^{-3}$ の値と、収束後の $10^{-4}$ 以下の値は MC の床と同程度で、0 と区別できない。

### 3.4 再構成された幾何(exp 4)

図: `sdf/K-1/W-{0.05,0.025}/learned_sdf_slice_<レンダラ>.pdf`(8 枚)。数値は `data/sdf/summary.log`。

| シーン | 真の厚さ | NA | NA_UP | BF | BF_UP |
|---|---|---|---|---|---|
| K = 1, W = 0.05 | 0.050 | 0.057 [0.056, 0.059] | 0.047 [0.046, 0.049] | 0.057 [0.055, 0.059] | 0.052 [0.051, 0.054] |
| K = 1, W = 0.025 | 0.025 | 0.046 [0.046, 0.048] | 0.043 [0.042, 0.043] | 0.046 [0.045, 0.047] | 0.043 [0.042, 0.044] |

(レイにわたる中央値と [25%, 75%]。消失したレイはどのランにもない。)

- **薄いスラブ(W = 0.025)は 4 レンダラすべてが約 2 倍に厚く再構成する。** 厚さ 0.043〜0.046 は標本間隔 Δ = 0.042 程度。真の面上の |SDF| の中央値は 0.009〜0.011 で、両面とも負(学習した面が外側)。つまり面が片側 0.01(画像の約 5 px。画素の足元での大きさは 0.0021)外に出ている。$\mathcal B^\uparrow$ を持つ 2 つは 0.003 だけ薄い。
- **標本間隔程度のスラブ(W = 0.05)では $\mathcal B^\uparrow$ の有無で差が出る。** NA と BF は 0.057(13% 厚い。面の |SDF| 0.003〜0.004)、NA_UP と BF_UP は 0.047 / 0.052(真値の 5% 以内。面の |SDF| 0.002)。近似 A の有無(NA 対 BF、NA_UP 対 BF_UP)は厚さに効かない。
- **断面図**: W = 0.025 では実線(学習)が破線(真値)の外側に平行に走り、W = 0.05 では $\mathcal B^\uparrow$ ありの 2 つでほぼ重なる。学習 SDF の勾配は面の外側で 0.93〜1.01 で、面の位置だけがずれている。面の実線の揺れは振幅 0.003 程度(x 方向を約 11 倍に引き伸ばしているので目立つ)。スラブの辺(y = ±0.5、z = ±0.5)から 0.03 の範囲では面が 0.01 外に膨らむ(どの高さでも同じ。床との接合ではない)。断面の y の範囲はスラブいっぱいなので、上下端にこの膨らみが見える。
- **9 × 9 のレイ(学習軌跡と同じ格子)でも中央値は 3 桁まで同じ**(格子の密度に依らない)。

### 3.5 結論

1. **Proposition 3.1 の誤差は、区間内のノイズの分散 $\Omega$ が定常分散より小さい範囲で $\Omega^2\propto\Delta t^2$ の速さで消える。** 実測の傾きは $\widetilde{\Omega}$ の掃引で 1.97($\widetilde{\Omega}<1$ の 5 点、参照が解像した 94 通りの中央値)、$\widetilde{\Delta t}$ の掃引で 2.34($\widetilde{\Delta t}<1$、85 通り)。$\widetilde{\Omega}=0.1$(κΔt ≈ 0.05)で相対誤差の中央値 0.3%、$\widetilde{\Omega}=1$(κΔt ≈ 0.35)で 3.8%、$\widetilde{\Delta t}=1$ で 13%。符号と係数は平均の曲率 $\tilde b$ と値 $\tilde\mu_0$ の組($\tilde b-\tilde\mu_0/2$)で決まり、始点の密度が平らな $\tilde\sigma=10$ で次数 2 が、境界が遠い $\tilde\sigma=0.1$ でそれより急な減衰が見える(3.1 節)。$\widetilde{\Omega}>1$ では誤差は消えず、$\widetilde{\Omega}\approx10$ で増加が止まる(中央値 1.6%、95% 点 0.43、最大 0.87)。$\widetilde{\Delta t}$ の掃引はその先($\widetilde{\Omega}$ で $5\times10^8$ まで)を覆い、$\widetilde{\Delta t}=10$ で中央値 0.38、95% 点 0.96。
2. **negative-absorbing 近似の失敗は、2 つの近似に分けてそれぞれの条件が特定できる。** 近似 A(survival の条件付けを落とす)は、複数の表面(K = 3)で、かつ過程の相関が強い(κ ≤ 1、τ ≤ 0.05)ときに破れる: 手前の表面で負になったパスを二重に数え、ヒット確率を 0.60 → 0.94 に過大評価する。表面が 1 枚なら効かない。近似 B($\mathcal B^\uparrow$ を落とす)は、表面が標本間隔より薄く(W = 0.6Δ)、かつ標本点の間に入っている(phase 0.5)ときに破れる: $\mathcal B^\uparrow$ を持つレンダラは表面を捉え、持たないレンダラは捉えない(ヒット確率 1 対 0)。表面が標本点を含めば(W = 1.2Δ、または phase 0.0)効かない。
3. **標本化そのものの限界が別にある。** 薄い表面が標本点の間にあり、かつノイズが小さい(τ ≲ 0.02)と、4 つのレンダラすべてが表面を見失う。これは近似 A・B の問題ではなく、標本点での値だけから交差を推定する枠組み(Proposition 3.1 の補間を含む)の限界で、$\mathcal B^\uparrow$ を入れても解決しない。
4. **実際の学習では、既定の手法(NA)の誤差は最初の数百ステップだけで、収束した場では 0 である。** 学習した場は κ が大きく τ が小さい、ほぼ決定的な過程に収束し、そこでは A も B も効かない。学習初期(step 100〜500)には Cramér 距離で 0.01〜0.05、ヒット確率の二乗距離で 0.03〜0.14 の誤差がある($\mathcal B^\uparrow$ だけを持つ NA_UP のヒット確率の過大評価が最大)。
5. **学習に沿った誤差が 0 になるのは、場が近似誤差を幾何の誤差に転化するからである。** 標本間隔より薄いスラブ(W = 0.6Δ)は、どのレンダラでも標本間隔程度(0.043〜0.046)に厚く再構成され、面が片側 0.01 外に出る。この設定では $\mathcal B^\uparrow$ の効果は 0.003 にとどまる。標本間隔程度のスラブ(W = 1.2Δ)では $\mathcal B^\uparrow$ ありが真値の 5% 以内、なしは 13% 厚い。近似 A は幾何に効かない。

### 3.6 査読への回答としての位置づけ

- **rXEt-R2(Proposition 3.1 の直接検証)**: 要求どおり、導出した確率(Eq. 26)を高解像度の MC(10⁸ パス、1000 細分割、外挿)と単一区間で直接比べた。振ったのは標本間隔($\widetilde{\Omega}$ で 2 桁 = κΔt で 0.05〜1.2 と、κΔt で 2 桁 = 0.1〜10 の 2 掃引)、OU の相関・ノイズと始点($\tilde\mu_0$、$\tilde\sigma$)、平均の形($\tilde a$、$\tilde b$)で、無次元化(2.2 節)によりこれで $(\kappa,\tau)$ の全域を覆う。答え: 誤差は $\widetilde{\Omega}<1$ で $\Omega^2$($\propto\Delta t^2$)の速さで消え(実測の傾き 2.0、κΔt の軸では 2.3)、その符号と係数は平均の曲率で決まる。$\widetilde{\Omega}>1$(区間内のノイズが定常分散を超える。κΔt ≳ 0.35)では消えず、$\widetilde{\Omega}\approx10$ で増加が止まる(中央値 1.6%、最大 0.87)。κΔt = 10 まで振ると中央値 0.38、最大 0.996(平均が区間内で負になって戻る薄い構造)。「複数回のゼロ交差の頻度」は、区間内で入って出る事象(up-crossing)そのものの確率で、掃引の中で 0 から 1 まで動く。
- **rXEt-R3(仮定を意図的に破るストレステスト)**: 要求どおり、複数表面(K = 3)と薄い構造(W = 0.6Δ)で、負になった後に正に戻る状況を作り、近似のあり・なしを同じ MC 参照と比べた。結論 2・3 が回答で、失敗の条件(表面の数と相関の強さ、厚みと標本点の配置)と機構(二重計上、$\mathcal B^\uparrow$ の欠落、標本化の限界)を示せる。実用上のトレードオフについては結論 4・5 が回答になる: 学習の誤差は初期だけで収束後は 0 だが、それは場が薄い構造を標本間隔程度に厚くして近似誤差を吸収するからで、代償は幾何の誤差(片側 0.01 = 約 5 px)として測れる(exp 4)。「$\mathcal B^\uparrow$ を入れれば消える」とは言えない(薄いスラブでは 0.003 の差)。言えるのは、標本間隔程度の構造では $\mathcal B^\uparrow$ ありが真値の 5% 以内、なしは 13% 厚い、という対比。
- **YmKF-R2(失敗例の議論の拡充。Critical)**: 複数の表面と交わるレイ(K = 3)での失敗例が、ヒット確率の過大評価(0.94 対 0.60)と first-passage 分布のずれ(Cramér 距離 0.05〜0.08)として定量的に示せる。複数回の下向き交差はこの設定そのもの(3 枚のスラブで 3 回の下向き交差)で、既定の手法が 2 回目以降を条件付けなしに数えることが失敗の原因である。結論 2〜5 が回答になる。学習における失敗の現れ方(幾何が標本間隔程度に太る)も具体例として挙げられる。
- 論文に載せるもの(予定): exp 1 の絶対誤差の図 1 枚(本文幅)、地図と学習に沿った誤差の 2×2 の組(位相・対・シーン・指標から選ぶ)。exp 4 の断面図(W = 0.025 の NA と BF_UP)と厚さの表を添える。上端の κΔt > 1 の行($\mathcal B^\uparrow$ ありの方が悪い。Proposition 3.1 の前提の外)と、地図の終点と学習誤差 0 の関係(exp 4 で説明)は、キャプションか本文で明記する(5 節)。

## 4. 再現方法

実行は `uv run python -m tools.analysis.<module>`(起動スクリプトは `bash tools/analysis/<name>.sh`)。重い計算はバッチに投入する: `bash outputs/TMLR-toy-analysis/jobs/submit.sh <name> <h_rt> <command...>`(`qsub -g $GROUP`、GPU 1 枚。ジョブのスクリプトとログは `jobs/<name>.{sh,log}` に生成される。完了したジョブのものは削除してあり、`jobs/` には `submit.sh` だけを残す)。全データは 2026-09-24 19:52〜22:55 の 192 ジョブ(exp 1 が 11 + 11、検算 2、地図 8、学習軌跡 160)で、`tools/analysis` を git add で固定した時点のコードにより最終設定(細分割 1000・間引き (1, 2, 4)・GL 1000)で 1 から計算した(同時実行 30 本で 3 時間)。MC の設定(パス数、細分割、間引き)と乱数シードは各評価器が `MonteCarloConfig`(`evaluate_approx_error.py`)で持ち、コマンドラインでは `--monte-carlo.<項目>` で指定し、出力の各記録の `config.monte_carlo` と `config.random_seed` に記録される。1 ジョブの所要時間は 4 節末尾の表のとおり。`summarize_approx_error.sh` が 3 節の数値と exp 1 のデータの検算を計算する(出力は `data/approx-errors/summary.log`)。

| 内容 | 評価 | 描画 |
|---|---|---|
| exp 1 | `bash tools/analysis/evaluate_up_cross_prob.sh <normalized_quadratic_variation または normalized_sampling_interval> <値>`(1 本につき 1 点。省略時は両掃引の 11 点すべて)、検算は `check_invariance.py` と `check_quadrature.py`(下のコードの項) | `bash tools/analysis/plot_up_cross_prob.sh` |
| 誤差地図 | `bash tools/analysis/evaluate_approx_error.sh <phases> <scenes>`(省略時は両位相・4 シーン。1 本 = 1 位相 × 1 シーン) | `bash tools/analysis/plot_approx_error.sh <phases> <フォルダの正規表現>`(地図と、学習に沿った誤差の図の両方を描く。指標ごとの共通レンジを全データから計算して渡す) |
| exp 4 | `bash tools/analysis/evaluate_sdf_error.sh <scenes> <variants>`(K = 1 の 2 シーン × 4 レンダラ、対話ノードの GPU で 1 ラン 1〜2 分)、表は `bash tools/analysis/summarize_sdf_error.sh` | `bash tools/analysis/plot_sdf_error.sh <scenes> <variants>` |
| 学習に沿った誤差 | `bash tools/analysis/evaluate_training_trajectory.sh <scenes> <variants> <slices>`(1 本 = 1 ラン × 10 チェックポイント。全 160 本)、完了後に `bash tools/analysis/merge_training_trajectory.sh` でランごとに結合 | 同上 |

- コード: `tools/analysis/`(git add で固定。exp 4 の再構成の道具 `evaluate_sdf_error.*`、`plot_sdf_error.*`、`summarize_sdf_error.*` を含む)。`evaluate_approx_error.py` に共通の関数(場の生成 `_create_field`、レンダラの切り替え `_configure_variant`、MC 参照 `_get_reference_cdf_values`、外挿の重み `_get_extrapolation_weights`、正規化 `_normalize_cdf_values`、距離 `_compute_metrics`、スラブ `CuboidConfig`)がある。exp 1 の評価器の入力は $\widetilde{\Delta t}=\kappa\Delta t$ で、$\widetilde{\Omega}$ の掃引は起動スクリプトが換算して渡す(2.2 節)。`check_quadrature.py` は exp 1 の式側の求積の節点数の根拠(2.2 節。`uv run python -m tools.analysis.check_quadrature --output-file outputs/TMLR-toy-analysis/data/approx-errors/checks/quadrature.json`。既定で両掃引の両端と中央を評価する)。`check_invariance.py` は無次元化の検算(2.2 節。`uv run python -m tools.analysis.check_invariance --output-file outputs/TMLR-toy-analysis/data/approx-errors/checks/invariance.json`。4 設定の記録は `checks/invariance/` に書く)。`plot_approx_error.py` に図の共通の様式(論文用の PDF、モデルの色と名前)がある。レンダラと MC のサンプラは `ssdp` の実装を import して使っている。
- データ: `outputs/TMLR-toy-analysis/data/approx-errors/`(exp 1〜3。`summary.log` もここ)、`data/sdf/`(exp 4: `<scene>_<variant>.{json,npz}` と `summary.log`)
  - `maps/phase-{0.0,0.5}/<scene>.json`: 121 格子点 × 4 レンダラの、MC との Cramér 距離(`conditional_cramer_distance`)、ヒット確率の二乗距離(`squared_distance`)、ヒット確率(`hit_prob_1` がレンダラ、`hit_prob_2` が MC)、距離の MC 誤差(`<指標>_stderr`、`<指標>_bias`、`hit_prob_2_stderr`。2.1 節)。各格子点の α と γ も記録してある
  - `trajectories/<scene>_<variant>.json`: チェックポイントごとの学習されたパラメータと、距離・その MC 誤差(2.1 節)・ヒット確率のレイにわたる平均・分位(ヒット確率で選別していないので、平均と上位の分位は外れ値の影響を受ける)。`trajectories/slices/` はジョブごとの出力(10 チェックポイント)で、結合の元
  - `up_cross_prob/normalized_quadratic_variation/<Ω̃>.json`、`up_cross_prob/normalized_sampling_interval/<κΔt>.json`(掃引ごとに 11 ファイル): exp 1。各記録に `normalized_quadratic_variation` と `normalized_sampling_interval` の両方がある
  - `checks/quadrature.json`: GL の節点数の確認(2.2 節)。`checks/invariance.json`: 無次元化の検算(2.2 節。4 設定の記録は `checks/invariance/`)
- 図: `outputs/TMLR-toy-analysis/figures/`(全 332 枚 = interval 2 掃引 × 2 + ray 2 位相 × 5 対 × 4 シーン × 2 指標 × 4 枚 + sdf 2 シーン × 4 レンダラ。2026-09-25 に固定したコードで空の状態から再生成)

```
interval/axis-{normalized_quadratic_variation,normalized_sampling_interval}/analytic_up_cross_prob_{abs,rel}_error_plot_vs_MC.pdf   横軸ごと(2.2 節)
ray/phase-{0.0,0.5}/
  approx-A/{BF-UP_vs_NA-UP,BF_vs_NA}/W-{0.05,0.025}/K-{1,3}/
  approx-B/{BF-UP_vs_BF,NA-UP_vs_NA}/K-{1,3}/W-{0.05,0.025}/
  approx-A+B/BF-UP_vs_NA/K-{1,3}/W-{0.05,0.025}/
    learned_first_passage_pmf_<指標>_distance_plot_<なし>_vs_<あり>_vs_MC.pdf   (位相に依らないので両方の枝に同じもの)
    analytic_first_passage_pmf_<指標>_distance_map_<レンダラ>_vs_MC.pdf   (2 枚)
    analytic_first_passage_pmf_<指標>_difference_map_<なし>_vs_<あり>.pdf
    (<指標> は cramer または squared)
sdf/K-1/W-{0.05,0.025}/learned_sdf_slice_<レンダラ>.pdf   学習した SDF の z = 0 断面(exp 4)
```

末端フォルダの 4 枚(1 指標分)を 2×2 で並べる予定。

1 ジョブの所要時間(バッチノードの GPU 1 枚、最終設定):

| ジョブ | 内容 | 所要時間 |
|---|---|---|
| exp 1(11 + 11 本) | 1 点、144 通り、$10^8$ パス、1000 細分割 | 16〜19 分 |
| 検算(2 本) | 無次元化: 4 設定 × 72 通り、$10^6$ パス。求積: 6 点 × 144 通り × 4 節点数 | 2〜3 分 |
| 地図(8 本) | 1 位相 × 1 シーン、121 格子点、$10^5$ パス、1000 細分割 | 29〜32 分 |
| 学習(160 本) | 1 ラン × 10 チェックポイント、946〜1930 レイ × $10^3$ パス、1000 細分割 | K = 1: 18〜23 分、K = 3: 36〜47 分 |
| exp 4(8 ラン、対話ノード) | 6561 レイ × 101 点 + 面 2 × 6561 点 + 断面 1001² | 1〜2 分 |


## 5. 未決定事項

図のレンジと目盛(2026-09-25 に決定: レンジはデータから、目盛は matplotlib の自動選択、端は目盛に合わせる)を含め、残っている未決定事項はない。

本文・キャプションで説明する事項(図の問題ではない):

- W = 0.6Δ の phase 0.5 の地図では軌跡の終点(κ 大、τ 小 = 左上)が距離の大きい領域にあるが、隣に並べる学習に沿った誤差は 0 である。理由は exp 4(3.4 節): 場がスラブを標本間隔程度に厚くして、自分のレンダラが見える形にしている。
- 地図の上端(κΔt > 1)の右上(τ ≥ 1)では 4 つのレンダラすべてが MC から離れ、$\mathcal B^\uparrow$ を持つ 2 つはさらに離れる。Proposition 3.1 の前提(区間が相関長より短い)の外で Eq. (26) が交差を過大に入れるためで、exp 1 の κΔt > 1 と同じ機構。学習した場は κ ≥ 46 に到達する(最大 112)のでこの領域を地図から外すことはできず、そのまま載せる。学習した場はこの角(τ ≥ 1)には入らない。
