# Phase C verdict

**VERDICT: FAIL**

No surviving instrument beat the better fixed choice on two or more
temporally distant fit/evaluate pairs.

**This is 'no effect detected', not 'no effect exists'** (FR-018d).
W3's marginal set holds 28 trades, which supports
detecting only a large effect; and outcomes are truncated near +3% by
the exit policy, compressing any magnitude signal.

## Evidence

| instrument         | level   | fit   | eval   | rule                                  |   n_marginal |   n_taken |   R_agg_pct |   R_con_pct |   R_phase_pct |   bar_pct |   edge_pct | beats_bar   |
|:-------------------|:--------|:------|:-------|:--------------------------------------|-------------:|----------:|------------:|------------:|--------------:|----------:|-----------:|:------------|
| btc_vol            | btc     | W1    | W3     | High Vol|Low Vol                      |           28 |        19 |      26.959 |      -2.088 |        18.831 |    26.959 |     -8.128 | False       |
| btc_vol            | btc     | W3    | W1     | High Vol|Low Vol|Mid Vol              |           46 |        46 |      22.789 |      11.774 |        22.789 |    22.789 |      0     | False       |
| btc_vol            | btc     | W2    | W3     | High Vol|Low Vol|Mid Vol              |           28 |        28 |      26.959 |      -2.088 |        26.959 |    26.959 |      0     | False       |
| btc_vol            | btc     | W3    | W2     | High Vol|Low Vol|Mid Vol              |           58 |        58 |      25.617 |      13.136 |        25.617 |    25.617 |      0     | False       |
| asset_vr4          | asset   | W1    | W3     | Neutral|Reverting                     |           28 |        25 |      26.959 |      -2.088 |        25.605 |    26.959 |     -1.354 | False       |
| asset_vr4          | asset   | W3    | W1     | Neutral|Reverting|Trending            |           46 |        46 |      22.789 |      11.774 |        22.789 |    22.789 |      0     | False       |
| asset_vr4          | asset   | W2    | W3     | Neutral|Reverting                     |           28 |        25 |      26.959 |      -2.088 |        25.605 |    26.959 |     -1.354 | False       |
| asset_vr4          | asset   | W3    | W2     | Neutral|Reverting|Trending            |           58 |        58 |      25.617 |      13.136 |        25.617 |    25.617 |      0     | False       |
| asset_vr24         | asset   | W1    | W3     | Reverting|Trending                    |           28 |        22 |      26.959 |      -2.088 |        22.878 |    26.959 |     -4.081 | False       |
| asset_vr24         | asset   | W3    | W1     | Neutral|Reverting                     |           46 |        35 |      22.789 |      11.774 |        17.068 |    22.789 |     -5.721 | False       |
| asset_vr24         | asset   | W2    | W3     | Reverting|Trending                    |           28 |        22 |      26.959 |      -2.088 |        22.878 |    26.959 |     -4.081 | False       |
| asset_vr24         | asset   | W3    | W2     | Neutral|Reverting                     |           58 |        48 |      25.617 |      13.136 |        20.579 |    25.617 |     -5.038 | False       |
| asset_hurst        | asset   | W1    | W3     | Anti-persistent|Persistent|Random     |           28 |        28 |      26.959 |      -2.088 |        26.959 |    26.959 |      0     | False       |
| asset_hurst        | asset   | W3    | W1     | Anti-persistent|Persistent|Random     |           46 |        46 |      22.789 |      11.774 |        22.789 |    22.789 |      0     | False       |
| asset_hurst        | asset   | W2    | W3     | Anti-persistent|Random                |           28 |        27 |      26.959 |      -2.088 |        26.458 |    26.959 |     -0.501 | False       |
| asset_hurst        | asset   | W3    | W2     | Anti-persistent|Persistent|Random     |           58 |        58 |      25.617 |      13.136 |        25.617 |    25.617 |      0     | False       |
| asset_autocorr     | asset   | W1    | W3     | Neg Autocorr|Pos Autocorr             |           28 |        14 |      26.959 |      -2.088 |        18.045 |    26.959 |     -8.915 | False       |
| asset_autocorr     | asset   | W3    | W1     | Neg Autocorr|No Autocorr|Pos Autocorr |           46 |        46 |      22.789 |      11.774 |        22.789 |    22.789 |      0     | False       |
| asset_autocorr     | asset   | W2    | W3     | Neg Autocorr|Pos Autocorr             |           28 |        14 |      26.959 |      -2.088 |        18.045 |    26.959 |     -8.915 | False       |
| asset_autocorr     | asset   | W3    | W2     | Neg Autocorr|No Autocorr|Pos Autocorr |           58 |        58 |      25.617 |      13.136 |        25.617 |    25.617 |      0     | False       |
| asset_trend        | asset   | W1    | W3     | Below EMA                             |           28 |        24 |      26.959 |      -2.088 |        25.903 |    26.959 |     -1.057 | False       |
| asset_trend        | asset   | W3    | W1     | Above EMA|Below EMA                   |           46 |        46 |      22.789 |      11.774 |        22.789 |    22.789 |      0     | False       |
| asset_trend        | asset   | W2    | W3     | Above EMA|Below EMA                   |           28 |        28 |      26.959 |      -2.088 |        26.959 |    26.959 |      0     | False       |
| asset_trend        | asset   | W3    | W2     | Above EMA|Below EMA                   |           58 |        58 |      25.617 |      13.136 |        25.617 |    25.617 |      0     | False       |
| market_breadth     | market  | W1    | W3     | Broad|Narrow                          |           28 |        28 |      26.959 |      -2.088 |        26.959 |    26.959 |      0     | False       |
| market_breadth     | market  | W3    | W1     | Narrow                                |           46 |        44 |      22.789 |      11.774 |        23.603 |    22.789 |      0.814 | True        |
| market_breadth     | market  | W2    | W3     | Broad|Mixed|Narrow                    |           28 |        28 |      26.959 |      -2.088 |        26.959 |    26.959 |      0     | False       |
| market_breadth     | market  | W3    | W2     | Narrow                                |           58 |        55 |      25.617 |      13.136 |        16.612 |    25.617 |     -9.005 | False       |
| market_correlation | market  | W1    | W3     | Coupled|Decoupled|Mid Correlation     |           28 |        28 |      26.959 |      -2.088 |        26.959 |    26.959 |      0     | False       |
| market_correlation | market  | W3    | W1     | Coupled|Decoupled|Mid Correlation     |           46 |        46 |      22.789 |      11.774 |        22.789 |    22.789 |      0     | False       |
| market_correlation | market  | W2    | W3     | Coupled|Decoupled                     |           28 |        26 |      26.959 |      -2.088 |        26.122 |    26.959 |     -0.837 | False       |
| market_correlation | market  | W3    | W2     | Coupled|Decoupled|Mid Correlation     |           58 |        58 |      25.617 |      13.136 |        25.617 |    25.617 |      0     | False       |

## Marginal sets

- **W1**: 46 trades, summed +22.79%
- **W2**: 58 trades, summed +25.62%
- **W3**: 28 trades, summed +26.96%
