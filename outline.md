Thanks for the details. I’ll research a technical trading strategy for hourly charts that is suitable for longer-term investing in the cryptocurrency shown. The strategy will be based on technical indicators such as RSI and MACD and will be designed for a \$10,000 capital base with medium risk tolerance.

I’ll identify promising strategies that are automatable and well-suited for backtesting, and provide a detailed explanation of how each works.
I’ll get back to you shortly with the findings.


# Strategy Overview

This crypto strategy uses hourly-bar signals for long-term trades, relying on RSI and MACD with complementary filters to capture multi-day moves. RSI (14) gauges momentum: values **>70** imply overbought and **<30** oversold. MACD uses a 12-26-9 configuration, producing a **MACD line** (12-EMA minus 26-EMA) and a 9-period signal line. A **bullish signal** is when MACD crosses above its signal line (histogram turns positive); a **bearish signal** is the opposite. Combining RSI and MACD helps confirm entry/exit points: for example, a MACD bullish crossover with RSI emerging from oversold can signal a good buy opportunity. This dual-indicator approach filters noise: RSI catches extreme price swings while MACD confirms trend direction.

# Indicators & Parameters

* **RSI** – Period 14 (default), oversold level \~30 and overbought \~70. (Crypto is volatile, so some traders tighten to 25/75 or use RSI<40 and >60 thresholds.)
* **MACD** – Standard (12, 26, 9) on hourly charts. The MACD histogram crossing above 0 or the line crossing above signal indicates bullish momentum. (For more smoothing on longer swings, longer EMAs like 19/39/9 or using 4h bars can reduce noise.)
* **Trend Filter** – A longer-period moving average on a higher timeframe (e.g. 50–200-period daily EMA/SMA) to confirm overall trend. We only take longs when price is above this MA, biasing toward the uptrend. (Similarly, ADX(14) >25 can ensure a strong trend.)
* **Volatility Filter** – ATR(14) helps set stops/trailing stops. For example, stop-loss at \~1×ATR below entry or swing low, and a 2×ATR trailing stop once in profit.
* **Additional** – Optionally, filter by **ADX** for trend strength and **volume** (e.g. volume > EMA-volume) to avoid false breakouts.

# Entry Conditions (Long)

1. **Trend Bias:** Ensure higher-timeframe uptrend (e.g. daily price > 50-day EMA or ADX>25) before opening new longs.
2. **RSI Signal:** On the hourly chart, RSI has dipped below the oversold threshold (e.g. <30) and then turns up. A crossover back above 30 (or a more moderate level like 40) can serve as a buy trigger.
3. **MACD Confirmation:** At or near the RSI trigger, the MACD line should cross above its signal line (histogram >0), indicating building upward momentum. Optionally require MACD line rising or MACD histogram increasing as further confirmation.
4. **Entry:** Enter at the close of the hour candle when the above conditions align. This captures “buy low” pullbacks in the context of an uptrend.

*Example:* A long entry would be triggered if (on an hourly close) *RSI(14)* crosses above 30 *and* *MACD(12,26,9)* line crosses above the signal line, while price is above its daily 50 MA.

# Exit Conditions

1. **Profit Target:** A simple target is 2× the risk (e.g. if risking 2%, target \~4%). This 2:1 reward\:risk is common for medium-risk traders. Alternatively, take profit when RSI climbs above 70 (overbought) or when MACD generates a bearish crossover.
2. **Indicator Exit:** Close the position if RSI crosses back below a high threshold (e.g. falls from >70 toward 50) or if MACD line crosses below its signal. For example, exit when MACD → signal line cross signals bearish momentum.
3. **Trailing Stop:** After reaching a threshold (e.g. +1×ATR move), use a trailing stop (e.g. 2×ATR) to lock profits. This lets winners run while protecting gains.

*Example:* If entry risk (difference between entry and stop) is 2%, set a take-profit \~4%. If price reaches that or RSI>70, exit part/all of the position. Otherwise, trail a stop based on ATR to capture larger moves.

# Stop-Loss & Risk Management

* **Initial Stop:** Place a stop-loss just below the recent swing low (for longs) or a fixed % (e.g. 1.5–2%) below entry. A common technique is 1×ATR below entry price or below the identified support level.
* **Position Sizing:** Risk no more than \~2% of total capital per trade.  With \$10k capital, this is about \$200 at risk. Calculate position size so that (Entry – Stop) × Size ≈ \$200.  For example, if entry \$50 and stop \$49 (2% move), buy \$200/(\$50–\$49)=200 units.
* **Risk per Trade:** This aligns with a medium-risk profile. Using fixed risk per trade and disciplined stops ensures drawdowns remain controlled.

# Position Sizing Formula

A simple rule is:

* **Dollar risk per trade** = Capital × risk% (e.g. \$10,000×2% = \$200).
* **Shares/coins** = (Dollar risk) / (Entry price – Stop price).
  This ensures a max loss ≈2% if stop is hit.

# Additional Filters and Enhancements

* **ADX Filter:** Use ADX(14) >25 to only trade when a trend is strong, reducing whipsaws.
* **Volume Filter:** Confirm breakouts with volume above average (e.g. volume > EMA), which can validate momentum.
* **Multi-Timeframe Check:** Require the daily or 4-hour RSI/MACD to also indicate bullishness (e.g. daily RSI >50, MACD above signal) before taking hourly signals.
* **Volatility Adaptation:** If volatility is extremely high, tighten thresholds or skip signals. For example, widen RSI bands (e.g. use 25/75) when ATR is very large.
* **Divergence:** Look for bullish divergence (price makes lower low but RSI/MACD do not) as extra confirmation, though this is optional.

# Backtesting Framework

This strategy is easily coded in Python, PineScript, etc.  A backtester would loop over hourly bars and apply these logic steps:

1. Compute indicators each hour (RSI, MACD, MA, ATR, ADX).
2. On each new bar, check entry: `if (Price> daily_SMA) and (RSI_prev<oversold and RSI_now>oversold) and (MACD fast crossed above signal): enter long`.
3. Set a stop-loss and target as described. If price subsequently hits stop, exit. If target hit or exit signal (RSI>70 or MACD cross down), exit.
4. Size trades so that `(entry_price – stop_price) * position_size ≈ $200 (2% of $10k)`.
   Backtesting platforms like Backtrader, Freqtrade, or TradingView’s Pine can implement these rules straightforwardly. Ensure to test the strategy over historical crypto data, adjusting parameters (RSI levels, MA length) for robustness.

# Strategy Summary

| **Parameter / Rule**        | **Example Setting**           | **Notes**                                                              |
| --------------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| RSI period                  | 14                            | Default momentum oscillator (0–100). Oversold <30, overbought >70.     |
| RSI buy threshold           | 30 (or 25)                    | Enter long when RSI crosses above this level from below.               |
| RSI sell threshold          | 70 (or 75)                    | Exit when RSI crosses below this level from above, signaling pullback. |
| MACD settings               | (12, 26, 9)                   | Standard MACD. Buy on line ↑ signal (histogram >0).                    |
| Trend filter (daily EMA/MA) | 50 or 200                     | Only go long if price is above this (uptrend).                         |
| ADX filter (14)             | >25                           | Only trade if trend strength is high.                                  |
| Entry signal                | RSI cross up + MACD cross     | Hourly RSI oversold crossover plus MACD bullish crossover.             |
| Stop-loss                   | \~1–2% (or 1×ATR) below entry | E.g. just below recent swing low.                                      |
| Take-profit                 | \~2× risk (e.g. 4%) or RSI>70 | Partial exit at 2:1 R\:R. The rest at RSI>70 or MACD cross down.       |
| Position sizing             | Risk 2% of \$10k (\$200)      | Positions sized so \$ loss ≈2% if stop hit.                            |

This framework is designed for a medium-risk trader: controlled risk per trade (1–2%), use of stops, and multi-indicator confirmation. By using hourly signals with daily trend context, it aims to capture swing moves without overtrading. All elements should be backtested over sufficient data to validate performance (past results do not guarantee future returns). Careful optimization (without overfitting) and ongoing tuning are essential for practical success.

**Sources:** Established trading resources and backtests on RSI/MACD strategies were used to define indicators, thresholds, and risk guidelines. These ensure the strategy is grounded in technical analysis best practices.
