# **Prediction Market Correlation Detection**

## **Project Plan**

---

## **1\. The Predictivity Matrix**

### **1.1 What We're Building**

We want to build an N×N matrix M where N is the number of prediction market contracts we're analyzing. Each cell M\[i\]\[j\] represents a single number: **how strongly market i's trading behavior predicts market j's outcome**.

* M\[i\]\[j\] \= 0 means markets i and j appear to be completely independent  
* M\[i\]\[j\] close to 1 means market i's behavior is a strong predictor of market j  
* The matrix is NOT necessarily symmetric — market i might predict j without j predicting i

If this matrix is accurate, we can use it to find **parlays that are mispriced** — bets that combine two events and are priced as if they're independent, when actually they're correlated.

### **1.2 How Each Cell Is Computed**

Each cell M\[i\]\[j\] is a **weighted sum of 7 different pairwise scores**, each capturing a different kind of relationship between the two markets:

M\[i\]\[j\] \= w1 \* F1(i,j) \+ w2 \* F2(i,j) \+ w3 \* F3(i,j) \+ w4 \* F4(i,j)  
         \+ w5 \* F5(i,j) \+ w6 \* F6(i,j) \+ w7 \* F7(i,j)

where F1 through F7 are the 7 feature functions (described in Section 3), and w1 through w7 are the global weights we optimize (described in Section 4).

Each feature function produces its own N×N matrix. The final predictivity matrix is a weighted combination of all 7\. The weights are the same for every cell — one global weight vector \[w1, w2, ..., w7\] applied everywhere.

---

## **2\. Data Source**

### **2.1 Platform: Polymarket**

We will use **Polymarket** as our sole data source. It is the largest prediction market by volume, has a fully public API requiring no authentication for read-only data, and has thousands of resolved (closed) markets we can use as ground truth.

### **2.2 APIs We Need**

**Gamma API** (market metadata — no auth required):

* Base URL: `https://gamma-api.polymarket.com`  
* `GET /markets?closed=true&limit=100&offset=0` — paginate through all resolved markets  
* Each market returns: `id`, `question`, `slug`, `category`, `outcomes`, `outcomePrices`, `volume`, `startDate`, `endDate`, `closedTime`, `tags`, `clobTokenIds`  
* The `clobTokenIds` field links each market to its tradeable tokens, which we need for price history

**CLOB API** (price history — no auth required for read-only):

* Base URL: `https://clob.polymarket.com`  
* `GET /prices-history?market=<tokenId>&interval=max&fidelity=720`  
  * `market` \= the token ID from the Gamma API's `clobTokenIds` field  
  * `interval` \= `max` gets the full history; other options: `1w`, `1d`, `6h`, `1h`  
  * `fidelity` \= resolution in minutes. For resolved markets, the minimum working fidelity is 720 (12 hours). Daily data \= fidelity of 1440\.  
  * Returns array of `{t: timestamp, p: price}` objects  
* **Important limitation**: for resolved/closed markets, sub-12-hour granularity returns empty data. Daily snapshots (fidelity=1440) are reliable.

**Data API** (trade-level data — no auth required for public trades):

* Base URL: `https://data-api.polymarket.com`  
* `GET /activity?market=<conditionId>` — individual trades with timestamps, sizes, prices, and buy/sell side  
* This is what we need for volume spike analysis

### **2.3 Sampling Strategy**

**Start broad, category-agnostic.** Pull all resolved markets from the Gamma API. Filter to:

* Markets with total volume \> $10,000 (filters out dead/illiquid markets)  
* Markets with at least 14 days of active trading (need enough price history to compute meaningful features)  
* Markets where `clobTokenIds` is present (some older markets lack this)

This should yield somewhere in the range of 1,000-5,000 usable resolved markets.

For pairwise analysis, only consider pairs where the active trading periods overlap by at least 7 days. This is necessary because most of our features require overlapping timeseries data. With N=2000 markets, there are \~2 million possible pairs, but after the overlap filter this drops substantially.

We do NOT filter by category at this stage. The whole point is to discover unexpected cross-category correlations.

### **2.4 Data Collection Script Outline**

import requests, time, json

\# Step 1: Pull all resolved markets from Gamma  
def fetch\_all\_resolved\_markets():  
    markets \= \[\]  
    offset \= 0  
    while True:  
        resp \= requests.get("https://gamma-api.polymarket.com/markets",  
            params={"closed": "true", "limit": 100, "offset": offset})  
        batch \= resp.json()  
        if not batch:  
            break  
        markets.extend(batch)  
        offset \+= 100  
        time.sleep(0.5)  \# respect rate limits  
    return markets

\# Step 2: For each market, pull daily price history from CLOB  
def fetch\_price\_history(token\_id):  
    resp \= requests.get("https://clob.polymarket.com/prices-history",  
        params={"market": token\_id, "interval": "max", "fidelity": 1440})  
    return resp.json()  \# list of {t: timestamp, p: price}

\# Step 3: For each market, pull trade activity from Data API  
def fetch\_trade\_activity(condition\_id):  
    resp \= requests.get(f"https://data-api.polymarket.com/activity",  
        params={"market": condition\_id})  
    return resp.json()

---

## **3\. The 7 Features**

Each feature is a function that takes two markets (i, j) and returns a number between 0 and 1 (after normalization). Below, each feature is explained assuming no prior statistics knowledge.

**IMPORTANT: All 7 feature matrices must be normalized to \[0, 1\] before combining.** This ensures the weights are directly comparable. Normalization method: for each matrix, subtract the minimum value and divide by (max \- min).

---

### **Feature 1: Price Correlation (Early Window)**

**What it measures:** Do the prices of these two markets tend to move up and down together?

**Intuition:** Each market has a daily price (between $0 and $1) representing the market's belief about how likely an event is. If two markets' prices tend to rise on the same days and fall on the same days, they're correlated.

**How to compute it:**

Pearson correlation is a standard formula that measures how much two lists of numbers move together. Given two lists of daily prices (one per market) during their overlap period:

            Σ (a\_i \- mean\_a)(b\_i \- mean\_b)  
r \= ───────────────────────────────────────────────  
    sqrt(Σ (a\_i \- mean\_a)²) \* sqrt(Σ (b\_i \- mean\_b)²)

* r \= \+1 means they move perfectly together  
* r \= 0 means no relationship  
* r \= \-1 means they move perfectly opposite

**The "early window" twist:** We only use the FIRST 40% of the overlap period to compute this. The reason: if we used the full overlap, we'd be "peeking" at data we're trying to predict. By using only early data, we're asking "did early price co-movement predict later co-movement?" — which is a genuine prediction.

**Data needed:** Daily price timeseries from CLOB `/prices-history` endpoint.

**Python:** `scipy.stats.pearsonr(prices_a[:split], prices_b[:split])` gives you both the correlation value and a p-value (how confident we are it's not just noise).

After computing, take absolute value |r| since we care about the strength of relationship, not direction. Then normalize to \[0,1\] across all pairs.

---

### **Feature 2: Granger Causality**

**What it measures:** Does knowing yesterday's price of market i help you predict today's price of market j, beyond what j's own history tells you?

**Intuition:** Regular correlation just says "they move together." Granger causality says "one LEADS the other." If California temperature markets consistently move 24 hours before wildfire markets, that's Granger causality — the temperature market's price *predicts* the wildfire market's future price.

**How to compute it:**

The Granger test runs two regressions:

1. Predict j's price today using only j's own past prices  
2. Predict j's price today using j's past prices AND i's past prices

If adding i's history significantly improves the prediction, we say i "Granger-causes" j. The test outputs an F-statistic (bigger \= stronger causal signal) and a p-value (smaller \= more confident the signal is real).

**Important:** This feature is ASYMMETRIC. F(i→j) ≠ F(j→i). This is the only asymmetric feature, and it's what makes our final matrix asymmetric too. This is actually valuable: "California heat predicts wildfire markets" might be true while the reverse is weaker.

**Data needed:** Daily price timeseries from CLOB `/prices-history`.

**Python:** `statsmodels.tsa.stattools.grangercausalitytests(data, maxlag=3)` returns test results for different lag values. Use the F-statistic from the best lag.

Normalize across all pairs to \[0,1\].

---

### **Feature 3: Volume Spike Co-occurrence**

**What it measures:** When one market has an unusually busy trading day, does the other market also tend to have an unusually busy day?

**Intuition:** Even if two markets' prices aren't moving together yet, if the same pool of traders suddenly starts trading both of them on the same days, that's a leading indicator that they're connected. News might be hitting that affects both events, even before prices fully adjust.

**How to compute it:**

1. For each market, get the daily trading volume (sum of all trade sizes per day)  
2. Compute each market's average daily volume and standard deviation  
3. A "spike day" is any day where volume \> mean \+ 2\*std  
4. Count how many of market i's spike days are also spike days for market j  
5. Divide by total spike days for market i

F3(i,j) \= |spike\_days\_i ∩ spike\_days\_j| / |spike\_days\_i|

This is also asymmetric (spike overlap relative to i vs relative to j may differ).

**Data needed:** Trade-level data from Data API `/activity` endpoint, aggregated to daily volume.

Normalize to \[0,1\] across all pairs.

---

### **Feature 4: Same Category**

**What it measures:** Are these two markets in the same Polymarket category?

**Intuition:** Markets in the same category (e.g., both "US Politics" or both "Crypto") are more likely to be driven by the same underlying forces. This is the simplest and cheapest feature — just a binary flag.

**How to compute it:**

F4(i,j) \= 1 if market\_i.category \== market\_j.category, else 0

**Data needed:** The `category` field from the Gamma API market response. Already available from the initial data pull — no additional API calls.

No normalization needed (already 0 or 1).

---

### **Feature 5: Tag Jaccard Similarity**

**What it measures:** How much do these two markets' topic tags overlap?

**Intuition:** Polymarket assigns tags to markets (e.g., "elections", "fed", "nba", "climate"). Two markets might be in different categories but share tags that reveal a connection. This is finer-grained than same-category.

**How to compute it:**

Jaccard similarity measures overlap between two sets:

F5(i,j) \= |tags\_i ∩ tags\_j| / |tags\_i ∪ tags\_j|

* 1.0 \= identical tag sets  
* 0.0 \= no tags in common  
* 0.5 \= half of their combined unique tags are shared

**Data needed:** The `tags` field from the Gamma API. Already available.

Already in \[0,1\] range — no normalization needed.

---

### **Feature 6: Resolution Date Proximity**

**What it measures:** How close in time did these two markets resolve?

**Intuition:** Markets that resolve on the same day or within a few days are more likely to be about related real-world events. A market resolving Jan 20 and another resolving Jan 21 might both be about the same political event. Markets resolving 6 months apart are less likely to be related.

**How to compute it:**

Use an exponential decay function so that same-day resolution scores highest and the score drops off smoothly:

F6(i,j) \= exp(-|days\_between\_resolution| / 30\)

* Same day: exp(0) \= 1.0  
* 1 week apart: exp(-7/30) ≈ 0.79  
* 1 month apart: exp(-1) ≈ 0.37  
* 3 months apart: exp(-3) ≈ 0.05

The 30-day decay constant is a design choice. It could be tuned, but it's not a priority.

**Data needed:** The `closedTime` field from the Gamma API. Already available.

Already in \[0,1\] range.

---

### **Feature 7: Temporal Overlap Ratio**

**What it measures:** What fraction of these two markets' active lifetimes overlapped?

**Intuition:** Two markets that were both active for the exact same 3-month period had maximum opportunity for their traders and prices to interact. Two markets where only 2 days overlapped have very little shared data, so any computed correlation is unreliable. This feature helps the model downweight pairs where the other features are computed from very little data.

**How to compute it:**

overlap\_start \= max(start\_i, start\_j)  
overlap\_end \= min(end\_i, end\_j)  
overlap\_days \= max(0, overlap\_end \- overlap\_start)

F7(i,j) \= overlap\_days / min(duration\_i, duration\_j)

We divide by the shorter market's duration so that a 2-week market fully contained within a 6-month market still gets a high score.

**Data needed:** `startDate` and `endDate` (or `closedTime`) from Gamma API. Already available.

Already in \[0,1\] range.

---

## **4\. Finding the Weights (Hill Climbing)**

### **4.1 The Optimization Problem**

We have 7 weights \[w1, w2, ..., w7\] to find. These are real-valued numbers. The goal is to find the weight vector that makes our predictivity matrix best match the actual outcomes of resolved markets.

### **4.2 Objective Function**

For every pair (i, j) of resolved markets, we know the ground truth: did they both resolve Yes, both resolve No, or mixed? We encode this as:

actual\_correlation(i,j) \= 1 if same outcome (both Yes or both No)  
                        \= 0 if different outcomes

Our predicted score for each pair is:

predicted(i,j) \= w1\*F1(i,j) \+ w2\*F2(i,j) \+ ... \+ w7\*F7(i,j)

The objective function to MINIMIZE is the mean squared error:

MSE \= (1/P) \* Σ\_all\_pairs (predicted(i,j) \- actual\_correlation(i,j))²

where P is the total number of pairs.

**Note:** This MSE is cheap to compute. All 7 feature matrices are precomputed once. For any candidate weight vector, computing MSE is just matrix arithmetic — multiply each precomputed matrix by its weight, sum, compare to ground truth. This runs in well under a second even for 100K+ pairs.

### **4.3 Hill Climbing Strategy**

We use **first-choice hill climbing with random restarts and simulated annealing** (combining concepts from the course).

ALGORITHM:

1\. Initialize: w \= \[1/7, 1/7, ..., 1/7\] (equal weights)  
   Set temperature T \= 1.0, decay d \= 0.995

2\. Compute current\_score \= MSE(w)

3\. Repeat for MAX\_ITERATIONS:  
   a. Generate neighbor: pick a random weight index k  
      \- Add a small random delta: w\_k' \= w\_k \+ uniform(-0.05, 0.05)  
      \- Clamp to \[0, 1\]  
      \- Re-normalize so all weights sum to 1 (optional but clean)  
     
   b. Compute new\_score \= MSE(w')  
     
   c. If new\_score \< current\_score:  
        Accept: w \= w', current\_score \= new\_score  
      Else:  
        Accept with probability exp(-(new\_score \- current\_score) / T)  
     
   d. Decay temperature: T \= T \* d

4\. Random restarts: repeat steps 1-3 from multiple random starting  
   weight vectors (e.g., 50 restarts), keep the best result.

5\. Return the weight vector with the lowest MSE across all restarts.

**Why hill climbing instead of just linear regression?** Linear regression gives the closed-form optimal solution for this exact setup. Hill climbing is used here because (a) this is an AI class and local search is core material, and (b) we can later introduce nonlinear combinations where closed-form solutions don't exist. The plan is to present both: linear regression as a baseline, hill climbing as the main method, and compare them.

### **4.4 Connecting to Course Material**

This directly implements concepts from the local search lectures:

* **First-choice hill climbing**: generate one neighbor, accept if better  
* **Simulated annealing**: accept worse moves with decaying probability to escape local optima  
* **Random restarts**: run from multiple starting points to avoid getting trapped  
* **Objective function design**: weighted MSE with normalization

---

## **5\. Train-Test Split**

**Time-based split, not random.** This is critical.

* **Training set**: all markets that resolved before a cutoff date (e.g., before Oct 2024\)  
* **Test set**: all markets that resolved after the cutoff date (e.g., Oct 2024 onward)

Why time-based? Because in real usage, we'd be using historical data to predict correlations in future markets. A random split would let the model "peek" at future data during training, which is cheating.

The training set is used to:

1. Precompute all 7 feature matrices for training pairs  
2. Run hill climbing to find optimal weights

The test set is used to:

1. Precompute feature matrices for test pairs  
2. Apply the learned weights (without re-optimizing)  
3. Measure how well the predictivity matrix matches actual test outcomes

**Evaluation metric on test set:**

* MSE between predicted scores and actual co-resolution (same metric as training, but on unseen data)  
* Also compute the Pearson correlation between predicted scores and actual outcomes — this tells us if the matrix's *ranking* of pairs is correct, even if the magnitudes are off

---

## **6\. Mispricing Analysis**

### **6.1 How Parlays Work**

A parlay bet combines two events: "A happens AND B happens." If the events are independent, the fair price of the parlay is:

fair\_parlay\_price \= P(A) \* P(B)

where P(A) and P(B) are the market prices (implied probabilities) of each event.

### **6.2 How Correlation Creates Mispricing**

If events A and B are positively correlated, then:

true\_joint\_probability \> P(A) \* P(B)

The parlay is UNDERPRICED — you're getting a better deal than the market realizes. Our predictivity matrix M\[i\]\[j\] estimates how correlated two markets are. A high M\[i\]\[j\] suggests the naive parlay price is too low.

### **6.3 Quantifying the Mispricing**

For each pair (i,j) in the test set:

1. Get the market prices at some point before resolution: P(i) and P(j)  
2. Compute naive parlay price: P(i) \* P(j)  
3. Compute actual co-resolution rate among similar pairs (pairs with similar M\[i\]\[j\] scores in the training data)  
4. Mispricing \= actual co-resolution rate \- naive parlay price

If our model works, the top-ranked pairs (highest M\[i\]\[j\]) should have the largest positive mispricing — meaning the actual co-resolution rate significantly exceeds the naive product.

### **6.4 Concrete Example**

Suppose market A ("Fed cuts rates in March") is priced at $0.60 and market B ("S\&P 500 up 5% by April") is priced at $0.40.

* Naive parlay: 0.60 \* 0.40 \= 0.24 (24% implied probability both happen)  
* Our matrix says M\[A\]\[B\] \= 0.7 (strong predicted correlation)  
* Historical data for pairs with similar M scores shows actual co-resolution of 0.35 (35%)  
* Mispricing: 0.35 \- 0.24 \= 0.11 (11 percentage points)

A bettor paying 24 cents for a parlay that actually hits 35% of the time has an edge.

---

## **7\. Evaluation: How to Judge If It Worked**

### **7.1 The Model Itself**

| Test | What It Measures | Success Threshold |
| ----- | ----- | ----- |
| Test set MSE | Prediction accuracy on unseen data | Lower than a "random weights" baseline |
| Test set correlation | Does ranking match reality? | Statistically significant positive r (p \< 0.05) |
| Hill climbing vs. equal weights | Do optimized weights beat naive 1/7 each? | Any MSE improvement |
| Hill climbing vs. linear regression | Does search find a comparable solution? | Within 10% of closed-form MSE |

### **7.2 Feature Importance**

After hill climbing, examine the learned weights. This tells us which features actually matter:

* If w1 (price correlation) dominates, the model is basically "past correlation predicts future correlation" — true but not novel  
* If w2 (Granger causality) or w3 (volume spikes) has significant weight, we've found that behavioral signals add real predictive power beyond simple correlation  
* If w4-w7 (metadata features) have near-zero weight, they're not helping and could be dropped

### **7.3 The Mispricing Test (The Real Payoff)**

This is the most important evaluation:

1. Sort all test-set pairs by their predicted M\[i\]\[j\] score (highest first)  
2. Take the top 10%, top 25%, top 50%  
3. For each group, compute:  
   * Average naive parlay price: mean of P(i)\*P(j)  
   * Actual co-resolution rate: fraction that both resolved the same way  
4. **If actual co-resolution rate \> naive parlay price for the top groups, and the gap is larger for higher-ranked groups, the model found real mispricing.**

Present this as a table or chart showing that higher predicted correlation → larger gap between actual outcome rate and naive pricing.

### **7.4 Baselines to Compare Against**

* **Random baseline:** random weight vector → establishes floor  
* **Equal weights:** all weights \= 1/7 → does optimization help?  
* **Single feature:** only price correlation, all other weights \= 0 → do the extra features add anything?  
* **Linear regression:** closed-form optimal weights → how close does hill climbing get?

If hill climbing beats all baselines and the mispricing test shows a real gap, the project is a success.

---

## **Appendix: Statistics Concepts Used**

For reference, since statistical background is limited:

**Pearson correlation (r):** Measures linear relationship between two number lists. Ranges from \-1 to \+1. Computed by scipy.stats.pearsonr().

**Granger causality:** Tests if one timeseries helps predict another. Not true causation, just predictive power. Computed by statsmodels.tsa.stattools.grangercausalitytests().

**Jaccard similarity:** Measures overlap between two sets. |A∩B| / |A∪B|. Easy to compute manually.

**Mean squared error (MSE):** Average of (predicted \- actual)² across all data points. Standard way to measure prediction accuracy. Lower \= better.

**p-value:** Probability that a result happened by random chance. Below 0.05 generally means "statistically significant" (probably real, not noise).

**Normalization (min-max):** Rescales a list of numbers to \[0, 1\]. formula: (x \- min) / (max \- min)

