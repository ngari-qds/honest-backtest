# Design notes: a backtester that argues with you

Every number here comes from `scripts/run_study.py` and lives in `results/tables/`.
The market it runs on is **generated, not observed** — see §1 for why that is the right
choice for this repository and where it makes the demo understate the problem.

---

## 1. Why the demo market is synthetic

A backtesting engine can be judged two ways: by whether its accounting is correct, and
by whether the strategy it happens to be running made money. Only the first is a
property of the engine.

Running on generated data with a **planted, known** effect means the naive-versus-honest
gap can be checked against ground truth. The generator plants 3% of annual return per
unit of standardised momentum signal; the honest run recovers a 5.25% annualised gross
return from it. If the engine ever recovered *more* edge than was planted, that would be
a bug, and without a ground truth there would be no way to notice.

Real prices would make the demo look more impressive and mean less. There would be no
true answer to compare against, and the headline would become "my backtester says this
strategy earns X" — which is the genre of claim this repository exists to argue with.

Two consequences, stated up front:

- **The level of every number here is meaningless.** Do not quote 0.77 as a Sharpe
  anyone achieved. The *gap* between the configurations is the result.
- **`is_synthetic` is a required field on `MarketData`**, not a default, and it
  propagates into every result object, every table and every chart subtitle. A chart is
  the part of a repository most likely to be screenshotted without its README, so the
  caveat has to survive the crop.

The data-generating process is written out in `data/synthetic.py`. The planted effect
depends only on strictly past prices, so it is genuinely tradeable rather than
contemporaneous — a contemporaneous effect would be unexploitable and would make the
honest run look wrongly bad.

## 2. Point-in-time discipline: making misuse difficult, not just documented

The usual way a backtest lies is that somebody aligns a signal to a return and gets the
shift backwards. Documenting the convention does not prevent this. The person who makes
that mistake is the person who did not read the docstring.

So the caller is not given the choice:

| what a caller might get wrong | how the API removes it |
|---|---|
| pairing a signal with the wrong return | the caller never supplies returns; the engine derives them from `MarketData` |
| choosing the alignment | there is no alignment parameter; a `Signal`'s row `t` is known at the close of `t`, full stop |
| forgetting the reporting lag | the engine applies it, from config |
| filling on the signal date | the engine executes on the *next* bar, always |
| a signal that peeks internally | `Signal.from_builder` runs a causality check and refuses to construct if it fails |

`assert_causal` is the general form of a look-ahead test: recompute the signal on data
whose future has been overwritten and require every past value to be unchanged. It
assumes nothing about *how* the signal is computed, so it keeps working when the
implementation changes — which a hand-written alignment assertion does not.

The test suite checks that it catches both the blatant mistake (`shift(-1)`) and the
realistic one: standardising against a **full-sample mean**, which contaminates every
date and is a mistake people make constantly.

### Something the control test revealed

The suite includes an oracle strategy that peeks at future prices, asserted to earn far
more than an honest one. Without that control, every other look-ahead assertion could
pass on an engine too blunt to notice anything.

Writing it exposed something worth keeping: **a one-day peek is destroyed by the
engine's own delays.** With a one-business-day reporting lag and next-bar execution, an
oracle that knows tomorrow's close cannot act on it in time, and its Sharpe came out
*worse* than the honest strategy's. The oracle had to be given a thirty-day horizon
before it could cheat measurably.

The engine being hard to cheat is the point. It also made the naive version of the
control useless, and the constant is named `PEEK_HORIZON_DAYS` with a comment saying why.

## 3. The accounting

### Split-day returns

On an ordinary day the book earns `weights · close-to-close return`. On a rebalance day
filled at the open, the day is split:

```
return = old_weights · (open/prev_close − 1) + new_weights · (close/open − 1)
```

You held yesterday's book overnight and today's from the opening print. Charging the
whole day to the new weights credits the strategy with an overnight move it had not yet
positioned for — negligible on any one day, systematic over 3,780 of them. This is the
only reason the engine takes open prices at all.

### Costs

| component | form | note |
|---|---|---|
| commission | `rate × traded notional` | per side |
| spread | `half_spread × traded notional` | crossing the quote once costs half the quoted spread, which is why the config says `half_spread_bps` rather than hiding a factor of two |
| impact | `coefficient × √participation × traded notional` | participation = traded shares ÷ trailing ADV, per name per day |
| borrow | `annual_fee ÷ 252 × short notional` | accrued **daily**, not on rebalance dates — it is the holding that costs money |

A name with no ADV history is charged at the configured participation cap rather than at
zero. Being unable to measure liquidity is not evidence that a trade was cheap.

### Conservation

`net = gross − commission − spread − impact − borrow` is asserted, not assumed. If a
cost component is ever added without being wired into the total, `attribute()` raises
rather than letting the discrepancy appear as alpha. The engine's own output is tested
to conserve **exactly** — to the bit, not approximately.

### The weight cap wins

If a leg has too few names for the per-name cap to be satisfiable — five names cannot
add up to 100% at a 5% cap — the book is left **under-invested** rather than the cap
being quietly breached to hit a gross target. A backtest that violates its own
concentration limit is reporting a portfolio nobody would have been allowed to hold.

## 4. Results

### The headline

| statistic | naive | honest | remaining |
|---|---|---|---|
| Sharpe | 1.278 | **0.773** | 60.5% |
| annualised return | 6.02% | **3.60%** | 59.8% |
| annualised volatility | 4.66% | 4.72% | — |
| max drawdown | −8.73% | −10.27% | 17.6% worse |
| annualised turnover | 5.65× | 5.76× | — |
| mean blocked shorts per day | 0 | 0.13 names | — |

### The ladder: which assumption was doing the work

Protections re-enabled one at a time, starting from fully naive:

| step | enabled | Sharpe | change |
|---|---|---|---|
| 0 | fully naive | 1.278 | — |
| 1 | charge commission, spread, impact | 1.126 | **−0.151** |
| 2 | charge stock borrow | 0.939 | **−0.188** |
| 3 | refuse hard-to-borrow shorts | 0.947 | +0.008 |
| 4 | apply the reporting lag | 1.051 | +0.104 |
| 5 | respect universe membership | 0.767 | **−0.284** |
| 6 | honour delistings | 0.773 | +0.007 |

**The single largest hit is survivorship, not costs.** Restricting the strategy to each
date's actual universe membership costs 0.284 of Sharpe by itself — nearly twice what
all three trading costs take together.

Two rungs move the wrong way and are reported as they came out rather than reordered to
look monotonic. The reporting lag *helping* by 0.104 is a real property of this sample —
a one-day delay happened to sidestep some short-term reversal — and it is a useful
reminder that these effects interact. **The ladder is a decomposition, not six
independent penalties**, and a different ordering would apportion the interactions
differently.

### Cost attribution

| component | annualised | share of gross |
|---|---|---|
| gross return | 5.25% | 100% |
| commission | −0.12% | 2.2% |
| spread | −0.46% | 8.8% |
| market impact | −0.14% | 2.6% |
| **borrow** | **−0.89%** | **17.0%** |
| net return | 3.65% | 69.4% |

**Borrow alone costs more than all three trading costs combined** (0.89% against 0.72%).
That is the least intuitive number in the repository, and it is a direct consequence of
charging borrow daily on held notional rather than only when trading.

### Capacity, and why the table does not believe itself

| quantity | value |
|---|---|
| assumed notional | $100m |
| annualised impact cost | 0.14% |
| estimated capacity | $36.7bn |
| implied scaling multiple | 367× |
| mean participation | 1.0% of ADV |
| 99th percentile participation | 13.1% of ADV |
| **max participation** | **121% of ADV** |
| trades over the 10% participation cap | 1.5% |
| **`capacity_estimate_is_credible`** | **False** |

The capacity model says $36.7bn. It is wrong, and the table says so. At $100m the
strategy already tries to trade 121% of one name's average daily volume on its worst
day, and 1.5% of trades breach the configured participation cap. Scaling 367× would
require a liquidity assumption that the same run has already falsified.

The participation tail is an *observation*; the capacity figure *extrapolates* a fitted
square-root law far past where it was calibrated. When the two disagree, believe the
observation. The flag exists so the table criticises itself rather than relying on the
reader to notice — a capacity number quoted without its participation distribution is
close to meaningless.

## 5. What this engine does not do, and what would mislead you

**Delistings in the synthetic market are random**, uncorrelated with past performance.
So terminal losses wash out between the long and short legs, which is why that rung of
the ladder barely moves (+0.007). In a real market delistings concentrate among past
losers, so a momentum short leg would collect them and the effect would be asymmetric
and much larger. **This is the single most important respect in which the demo
understates the problem it exists to demonstrate.**

**Borrow tiers are assigned randomly.** Real borrow cost correlates with short interest
and float, and the names a momentum strategy wants to short are disproportionately the
expensive ones. The borrow line is the right order of magnitude attached to the wrong
names, which means the true cost is understated.

**The impact coefficient is invented.** The square-root form is standard; the value of
10 is not calibrated against fills. It sets the scale of the impact column.

**No intraday modelling, no partial fills, no queue position, no shorting recall.**
Trades execute in full at the open or the close. This is a daily-bar cross-sectional
engine and it does not pretend otherwise. It is explicitly not an event-driven engine,
an order router, or a live trading system.

**The ladder's ordering is a choice.** A Shapley decomposition would attribute the
interactions fairly and is not implemented.

**Weekly rebalancing is supported but untested at scale.** The demo runs monthly.

---

## What I would trust this for, and what I would not

I would trust it to answer *"how much of this backtest is an artefact of its own
assumptions?"* — that is what it was built for, the accounting is asserted rather than
assumed, and the ladder attributes the answer to specific causes.

I would not trust it to produce a number anyone should size a position on. It runs on
daily bars, fills in full, invents its impact coefficient, and — in the demo — runs on a
market I generated. Every one of those is fine for measuring the *gap* between two sets
of assumptions and none of them is fine for measuring an expected return.

The honest summary of the repository is narrow and I would rather state it than dress it
up: **on a strategy with a genuine, planted edge, roughly 40% of the apparent Sharpe was
an artefact of assumptions a careless backtest makes by default — and more of that came
from survivorship than from trading costs.**
