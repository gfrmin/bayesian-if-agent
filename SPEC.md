# Bayesian IF Agent: Specification v6

## Design Principles (NON-NEGOTIABLE)

These principles are not suggestions. They define what this project is.

### 1. Everything is Expected Utility Maximisation

Every decision — whether to ask a question or take a game action — is:

$$a^* = \arg\max_a \mathbb{E}[U | a, \text{beliefs}]$$

No special cases. No "follow LLM 70% of the time." No "explore with probability ε." Just EU.

### 2. No Hacks

If the agent behaves badly, the solution is better modeling, not bolted-on fixes.

**Forbidden:**
- Exploration bonuses
- Loop detection
- Novelty rewards
- "If stuck, do X"
- Any rule that isn't derived from EU maximisation

**Instead:** Figure out why EU maximisation gives wrong answer. Fix the model.

### 3. LLM Outputs Are Data

The LLM is a sensor. Its outputs are observations that update beliefs via Bayes' rule:

$$P(\text{true} | \text{LLM says yes}) = \frac{P(\text{LLM says yes} | \text{true}) \cdot P(\text{true})}{P(\text{LLM says yes})}$$

The agent learns sensor reliability (TPR, FPR) from experience. If the LLM is unreliable, the agent learns to ignore it.

LLM outputs are **never** commands to follow blindly.

### 4. Direct Experience Is Certain

The game is deterministic. If we tried action A in state S and got outcome O:

$$P(O | S, A) = 1$$

No uncertainty. Known outcomes have EU = observed reward − c_act. No need to ask the LLM about known outcomes.

### 5. Be Honest About Parameters

Every parameter must be justified:
- **Question cost c:** Real cost of computation time. Not a hack to limit questions.
- **Action cost c_act:** Opportunity cost of spending a game turn. Derived from finite horizon.
- **Prior P(helps):** Beta(1/N, 1−1/N). Derived from problem structure, not a magic number.
- **Sensor priors (TPR, FPR):** Initial estimates, updated from data.

If we can't justify a parameter, we shouldn't have it.

### 6. When Stuck, Ask: What Would a Rational Agent Do?

Then derive the math. If the math says something surprising, either:
- Our model is missing something (fix the model)
- The result is actually correct (accept it)

Never: "The math says X but let's do Y instead."

---

## The Problem

An agent plays text adventure games (Interactive Fiction). It observes text, chooses from valid actions, receives rewards. Goal: maximise score.

The agent has access to an LLM oracle it can query with yes/no questions.

**Challenge:** Games require long action sequences with sparse rewards. Random exploration fails. The LLM has relevant knowledge but is imperfect.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         GAME                                │
│  - Observation (text)                                       │
│  - Valid actions                                            │
│  - Reward (sparse)                                          │
│  - State (deterministic)                                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AGENT BELIEFS                            │
│                                                             │
│  For each action a:                                         │
│    - P(a helps | evidence)                                  │
│    - Known outcome if tried before                          │
│                                                             │
│  For LLM sensor:                                            │
│    - TPR: P(says yes | actually helps)                      │
│    - FPR: P(says yes | actually doesn't help)               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 UNIFIED DECISION                            │
│                                                             │
│  Options:                                                   │
│    - take(A): Execute game action A                         │
│    - ask(Q): Query LLM with question Q                      │
│                                                             │
│  Choose: argmax EU                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Unified Decision Space

The agent chooses between:
- `take(A)` — execute game action, spend one turn, get reward
- `ask(Q)` — query LLM, spend computation time, update beliefs

### EU of Game Actions

Taking a game action costs a turn. In a finite-horizon game with T turns remaining and ~1 correct action per state, each wasted turn has expected opportunity cost. We encode this as $c_{\text{act}}$ — not a tuning knob, but a consequence of finite horizon. Without it, EU can't distinguish "ask a free question" from "burn a game turn."

For action A in state S:

**If outcome known** (we've tried A in S before):
$$\mathbb{E}[U | \text{take}(A)] = R_{\text{observed}} - c_{\text{act}}$$

**If outcome unknown:**
$$\mathbb{E}[U | \text{take}(A)] = P(A \text{ helps}) \times R_{\text{success}} - c_{\text{act}}$$

With $R_{\text{success}} = 1$:
$$\mathbb{E}[U | \text{take}(A)] = P(A \text{ helps}) - c_{\text{act}}$$

### EU of Asking

For question Q:
$$\mathbb{E}[U | \text{ask}(Q)] = \mathbb{E}[\max_A EU(A) | \text{after asking}] - c$$

Where:
- First term: expected best EU after belief update from answer
- c: cost of asking (computation time, in utility units)

### Value of Information

$$\text{VOI}(Q) = \mathbb{E}[\max_A EU(A) | \text{after}] - \max_A EU(A) | \text{before}$$

This is the expected improvement from asking.

The EU of asking is:
$$\mathbb{E}[U | \text{ask}(Q)] = \max_A EU(A)_{\text{before}} + \text{VOI}(Q) - c$$

### The Decision Rule

Compare best question to best game action:

$$\text{Ask if: } \text{VOI}(Q^*) > c$$

Where $Q^* = \arg\max_Q \text{VOI}(Q)$.

Equivalently: ask if the improvement from asking exceeds the cost.

---

## Computing VOI

For a question about action A:

1. **Current belief:** $b = P(A \text{ helps})$ (Beta mean: $\alpha / (\alpha + \beta)$)

2. **Predict LLM response:**
   $$P(\text{yes}) = \text{TPR} \times b + \text{FPR} \times (1-b)$$

3. **Posterior if yes:**
   $$b_{\text{yes}} = \frac{\text{TPR} \times b}{\text{TPR} \times b + \text{FPR} \times (1-b)}$$

4. **Posterior if no:**
   $$b_{\text{no}} = \frac{(1-\text{TPR}) \times b}{(1-\text{TPR}) \times b + (1-\text{FPR}) \times (1-b)}$$

5. **Best EU after each answer:**
   $$EU^*_{\text{yes}} = \max(b_{\text{yes}} - c_{\text{act}}, \max_{A' \neq A} EU(A'))$$
   $$EU^*_{\text{no}} = \max(b_{\text{no}} - c_{\text{act}}, \max_{A' \neq A} EU(A'))$$

6. **VOI:**
   $$\text{VOI} = P(\text{yes}) \times EU^*_{\text{yes}} + P(\text{no}) \times EU^*_{\text{no}} - EU^*_{\text{before}}$$

---

## Learning Sensor Reliability

The agent learns TPR and FPR from experience.

### Ground Truth

When we take an action and observe the outcome, we get ground truth:

$$\text{helped} = (\text{reward} > 0)$$

**Important:** Only reward counts. Not state change. Changing clothes changes state but doesn't help. Noisy ground truth poisons learning.

### Bayesian Update

Model TPR and FPR as Beta distributions:

```
TPR ~ Beta(α_tp, β_tp)
FPR ~ Beta(α_fp, β_fp)
```

Update rules:
- LLM said yes, actually helped: α_tp += 1
- LLM said yes, actually didn't help: α_fp += 1
- LLM said no, actually helped: β_tp += 1
- LLM said no, actually didn't help: β_fp += 1

Point estimates:
$$\text{TPR} = \frac{\alpha_{tp}}{\alpha_{tp} + \beta_{tp}}$$
$$\text{FPR} = \frac{\alpha_{fp}}{\alpha_{fp} + \beta_{fp}}$$

---

## Prior Over Actions

Each action's P(helps) is a Beta distribution, just like sensor TPR/FPR. This makes the prior **updateable** from both LLM observations and direct game outcomes.

$$P(A \text{ helps}) \sim \text{Beta}(\alpha_A, \beta_A)$$

Point estimate (for EU calculation): $\hat{p} = \alpha_A / (\alpha_A + \beta_A)$

**The probability itself is uncertain.** The total count $\alpha + \beta$ measures how sure we are about $\hat{p}$. A low total count means the mean will shift substantially with new evidence.

Default prior: **Beta(1/N, 1 − 1/N)** where N = number of available actions.

- **Mean 1/N:** derives from problem structure. In most IF states, roughly one action advances the game. With no other information, spread that probability uniformly. The LLM sensor exists to break this symmetry. For N=15 actions, mean ≈ 0.07. Pessimistic — but that's correct when you have 15 choices and one is right.
- **Total count 1:** maximally weak. A single ground truth observation doubles the total count and shifts the mean substantially. One "helped" → Beta(1/N + 1, 1 − 1/N), mean jumps dramatically. One "didn't help" → Beta(1/N, 2 − 1/N), mean drops toward zero.
- **Why 1/N and not a fixed constant?** A fixed constant like 0.15 is a magic number — it doesn't derive from anything. 1/N derives from the structure: one action helps, don't know which. No magic.

### Updating from ground truth

Exact conjugate update (strong evidence, total count grows by 1):
- Helped (reward > 0): $\alpha_A$ += 1
- Didn't help (reward ≤ 0): $\beta_A$ += 1

After 10 observations of "didn't help" starting from Beta(0.07, 0.93): Beta(0.07, 10.93), mean ≈ 0.006 — very certain it's bad.

### Updating from LLM observations

The LLM sensor gives a noisy binary observation about action A. The exact Bayesian update is:

$$P(\theta | \text{obs}) \propto P(\text{obs} | \theta) \, P(\theta)$$

where $P(\text{yes} | \theta) = \text{TPR} \cdot \theta + \text{FPR} \cdot (1 - \theta)$. This likelihood is linear in θ, so the posterior is **not** Beta — the conjugacy breaks. The implementation must approximate, e.g. by moment-matching back to a Beta. The spec does not prescribe a specific approximation; what matters is that the update is derived from Bayes' rule with the sensor's learned TPR/FPR, and that the result is a proper distribution (not a bare point estimate).

### Why not 0.5?

A Beta(1,1) prior (mean 0.5) says "each action is equally likely to help or not" — absurd for IF games. It overestimates EU for every action and suppresses VOI (less to learn when you're already confident).

### Three families of Beta distributions

After these changes, the agent holds exactly three families of Beta distributions:

1. **Sensor TPR** ~ Beta(α\_tp, β\_tp) — "how often does the LLM say yes when the action truly helps?"
2. **Sensor FPR** ~ Beta(α\_fp, β\_fp) — "how often does the LLM say yes when the action doesn't help?"
3. **Action belief** ~ Beta(α\_a, β\_a) per action — "does this action help?"

Every belief is a distribution. Every distribution learns. Every decision flows from these distributions via EU maximisation. Nothing else.

---

## Why No Exploration Bonus

With proper priors:
- Known bad action: EU = 0 − c_act < 0
- Unknown action: EU = P(helps) − c_act (positive when P(helps) > c_act)

Unknown actions already have higher EU than known-bad ones. Exploration emerges from uncertainty. No bonus needed.

---

## Why No Loop Detection

The agent might cycle through clothing states. Each (state, action) pair is technically new because clothing combinations create different states.

**The right fix:** The LLM should tell us "clothing actions don't help." If VOI > cost, we ask. The LLM says "no, putting on pants won't get you to work." We update P(helps) downward. We stop cycling.

**The wrong fix:** Detect loops, force random action. This is a hack that hides the real problem.

If the agent still loops after asking, either:
- The LLM is unreliable (we learn this, stop trusting it)
- The VOI calculation has a bug (fix the bug)
- The model is missing something (improve the model)

---

## Implementation

### BinarySensor

```python
@dataclass
class BinarySensor:
    """Yes/no sensor with learned reliability."""
    
    # TPR: P(yes | true)
    tp_alpha: float = 2.0
    tp_beta: float = 1.0
    
    # FPR: P(yes | false)  
    fp_alpha: float = 1.0
    fp_beta: float = 2.0
    
    @property
    def tpr(self) -> float:
        return self.tp_alpha / (self.tp_alpha + self.tp_beta)
    
    @property
    def fpr(self) -> float:
        return self.fp_alpha / (self.fp_alpha + self.fp_beta)
    
    def posterior(self, prior: float, said_yes: bool) -> float:
        """P(true | LLM response)"""
        if said_yes:
            num = self.tpr * prior
            denom = self.tpr * prior + self.fpr * (1 - prior)
        else:
            num = (1 - self.tpr) * prior
            denom = (1 - self.tpr) * prior + (1 - self.fpr) * (1 - prior)
        return num / denom if denom > 0 else prior
    
    def update(self, said_yes: bool, actual: bool):
        """Update from ground truth."""
        if actual:
            if said_yes:
                self.tp_alpha += 1
            else:
                self.tp_beta += 1
        else:
            if said_yes:
                self.fp_alpha += 1
            else:
                self.fp_beta += 1
```

### DynamicsModel

```python
class DynamicsModel:
    """Records observed outcomes. Deterministic game = one observation = certainty."""
    
    def __init__(self):
        self.observations: Dict[Tuple[str, str], float] = {}  # (state, action) -> reward
    
    def record(self, state: str, action: str, reward: float):
        self.observations[(state, action)] = reward
    
    def known_reward(self, state: str, action: str) -> Optional[float]:
        return self.observations.get((state, action))
    
    def is_known(self, state: str, action: str) -> bool:
        return (state, action) in self.observations
```

### UnifiedDecisionMaker

```python
class UnifiedDecisionMaker:
    """Choose between asking and acting via EU maximisation."""

    def __init__(self, question_cost: float, action_cost: float):
        self.question_cost = question_cost
        self.action_cost = action_cost

    def choose(
        self,
        state: str,
        actions: List[str],
        beliefs: Dict[str, Tuple[float, float]],  # action -> (α, β)
        sensor: BinarySensor,
        dynamics: DynamicsModel,
    ) -> Tuple[str, Any]:  # ('ask', action) or ('take', action)

        # Compute EU for each game action
        game_eus = {}
        n = len(actions)
        for a in actions:
            known = dynamics.known_reward(state, a)
            if known is not None:
                game_eus[a] = known - self.action_cost
            else:
                alpha, beta = beliefs.get(a, (1.0/n, 1.0 - 1.0/n))
                game_eus[a] = alpha / (alpha + beta) - self.action_cost
        
        best_game_eu = max(game_eus.values())
        best_game_action = max(actions, key=lambda a: game_eus[a])
        
        # Find best question (highest VOI)
        best_voi = 0.0
        best_question_action = None
        
        for a in actions:
            if dynamics.is_known(state, a):
                continue  # No point asking about known outcomes

            alpha, beta = beliefs.get(a, (1.0/n, 1.0 - 1.0/n))
            voi = self.compute_voi(a, alpha, beta, sensor, game_eus)
            if voi > best_voi:
                best_voi = voi
                best_question_action = a
        
        # Decision: ask if VOI > cost
        if best_question_action is not None and best_voi > self.question_cost:
            return ('ask', best_question_action)
        else:
            return ('take', best_game_action)
    
    def compute_voi(
        self,
        action: str,
        alpha: float,
        beta: float,
        sensor: BinarySensor,
        game_eus: Dict[str, float],
    ) -> float:
        """Value of information for asking about action.

        Uses Beta parameters (not just the mean) — two actions with the
        same mean but different total counts have different VOI because
        the uncertain one has more to learn from a question.
        """

        belief = alpha / (alpha + beta)
        current_best = max(game_eus.values())

        # P(LLM says yes)
        p_yes = sensor.tpr * belief + sensor.fpr * (1 - belief)

        # Posteriors
        post_yes = sensor.posterior(belief, True)
        post_no = sensor.posterior(belief, False)

        # EU of this action under each posterior (minus action_cost)
        eu_yes = post_yes - self.action_cost
        eu_no = post_no - self.action_cost

        # Best EU after each answer (other actions unchanged)
        other_best = max((eu for a, eu in game_eus.items() if a != action), default=0)
        best_if_yes = max(eu_yes, other_best)
        best_if_no = max(eu_no, other_best)

        # Expected best after asking
        expected_best = p_yes * best_if_yes + (1 - p_yes) * best_if_no

        return max(0.0, expected_best - current_best)
```

### Agent

```python
class BayesianIFAgent:
    """Bayesian IF agent. All decisions via EU maximisation.

    Three families of Beta distributions, all learning from data:
      1. Sensor TPR ~ Beta(α_tp, β_tp)
      2. Sensor FPR ~ Beta(α_fp, β_fp)
      3. Action belief ~ Beta(α_a, β_a) per action
    Every belief is a distribution. Every distribution learns.
    Every decision flows from these distributions via EU maximisation.
    """

    def __init__(self, llm_client, question_cost: float = 0.01,
                 action_cost: float = 0.10):
        self.llm = llm_client
        self.sensor = BinarySensor()
        self.dynamics = DynamicsModel()
        self.decision_maker = UnifiedDecisionMaker(question_cost, action_cost)

        self.beliefs: Dict[str, Tuple[float, float]] = {}  # action -> (α, β)
        self.pending_predictions: Dict[str, bool] = {}  # action -> LLM said yes

    def act(self, state: str, observation: str, actions: List[str]) -> str:
        """Choose and return action to take."""

        # Initialize beliefs for new actions
        # Prior: Beta(1/N, 1 - 1/N) — mean 1/N, total count 1 (maximally weak).
        # Derives from problem structure: ~1 action helps, don't know which.
        n = len(actions)
        for a in actions:
            if a not in self.beliefs:
                self.beliefs[a] = (1.0 / n, 1.0 - 1.0 / n)

        # Decision loop: ask or act?
        while True:
            decision, value = self.decision_maker.choose(
                state, actions, self.beliefs, self.sensor, self.dynamics
            )

            if decision == 'take':
                return value

            # Ask question
            action_to_ask = value
            question = f"Will '{action_to_ask}' help make progress? YES or NO."
            answer = self.llm.ask(question, observation)
            said_yes = 'YES' in answer.upper()

            # Update belief via Bayes' rule using sensor TPR/FPR,
            # then approximate posterior as Beta (see § Prior Over Actions).
            alpha, beta = self.beliefs[action_to_ask]
            self.beliefs[action_to_ask] = self.update_belief_from_sensor(
                alpha, beta, self.sensor, said_yes
            )

            # Record for ground truth learning
            self.pending_predictions[action_to_ask] = said_yes

    def observe(self, state: str, action: str, reward: float):
        """Learn from outcome."""

        # Record dynamics
        self.dynamics.record(state, action, reward)

        # Update sensor from ground truth
        helped = reward > 0  # Strict: only score counts

        if action in self.pending_predictions:
            said_yes = self.pending_predictions[action]
            self.sensor.update(said_yes, helped)
            del self.pending_predictions[action]

        # Update action belief from ground truth (exact conjugate update)
        if action in self.beliefs:
            alpha, beta = self.beliefs[action]
            if helped:
                self.beliefs[action] = (alpha + 1, beta)
            else:
                self.beliefs[action] = (alpha, beta + 1)
```

---

## Parameters

| Parameter | Value | Derivation |
|-----------|-------|------------|
| question_cost | 0.01 | ~1% of max reward. Asking is cheap but not free. |
| action_cost | 0.10 | Opportunity cost of a game turn. Derived from finite horizon: each wasted turn forfeits ~1/T of remaining achievable score. |
| prior P(helps) | Beta(1/N, 1−1/N) | Mean 1/N (one action helps, don't know which), total count 1 (maximally weak). Derived from problem structure. |
| TPR prior | Beta(2, 1) | Expect LLM is somewhat reliable (mean 0.67). |
| FPR prior | Beta(1, 2) | Expect LLM doesn't say yes to everything (mean 0.33). |

These can be tuned. But they must be justified, not arbitrary.

### Model Agnosticism

The Bayesian framework is model-agnostic — it learns any sensor's reliability from data. Default: any instruction-following LLM via Ollama (e.g. llama3.1 8b, 70b, etc). A larger model may start with better TPR/FPR, but the agent will learn either way. The math doesn't care what's behind the sensor.

---

## Debugging Checklist

If agent behaves badly:

1. **Is VOI being computed correctly?** Print VOI for each potential question.

2. **Is the comparison right?** Ask if VOI > cost, not if (VOI - cost) > game_eu.

3. **Are beliefs reasonable?** Print P(helps) for actions. Should be ~1/N initially (Beta prior mean).

4. **Is ground truth correct?** Only reward > 0 counts. Not state change.

5. **Is the LLM being queried?** If VOI > cost, questions should happen.

6. **What does the LLM say?** Maybe it's wrong. Check actual responses.

**Never:** Add a hack to fix symptoms. Find the root cause.

---

## Success Criteria

1. **Agent asks questions** when VOI > cost
2. **Beliefs update** from LLM answers
3. **Sensor reliability learned** from outcomes
4. **No loops** — if cycling, agent asks LLM and learns those actions don't help
5. **All behavior explainable** via EU maximisation

---

## Categorical Suggestion Sensor

### Motivation

The binary "will action X help?" question has two problems:
1. One question per action → N questions to assess N actions
2. TPR ≈ 0 in practice — LLMs say YES to everything

A categorical question — "which action should I take?" — addresses both. One question informs beliefs about all N actions simultaneously.

### Model

**Question:** Present numbered action list, ask LLM to pick one.

**Accuracy:** Scalar $a \sim \text{Beta}(\alpha, \beta)$, prior Beta(2, 1) (mean 0.67).

**Likelihood:**
$$P(\text{LLM suggests } i \mid \text{action } j \text{ is correct}) = \begin{cases} a & \text{if } i = j \\ \frac{1-a}{N-1} & \text{if } i \neq j \end{cases}$$

**Posterior via Bayes' rule:**
$$P(\text{action } j \text{ correct} \mid \text{LLM suggests } i) = \frac{P(\text{suggests } i \mid j) \cdot P(j)}{\sum_k P(\text{suggests } i \mid k) \cdot P(k)}$$

**Moment-matching:** The categorical posterior is converted back to Beta parameters for each action by preserving total count (incremented by 1) and setting the mean to the posterior.

### VOI for Categorical Suggestion

$$\text{VOI}_{\text{cat}} = \sum_{i} P(\text{suggests } i) \cdot \max_j \left[ P(j \mid \text{suggests } i) - c_{\text{act}} \right] - \max_j \left[ P(j) - c_{\text{act}} \right]$$

Where $P(\text{suggests } i) = \sum_j P(\text{suggests } i \mid j) \cdot P(j)$.

This competes with binary VOI in the unified decision framework. In practice, categorical dominates because one question updates all N actions.

### Ground Truth

- **Reward > 0** after taking suggested action → correct
- **Reward > 0** after taking different action → incorrect
- **Reward == 0** → consult progress sensor (see below)

---

## Progress Evaluator

### Role

Supplementary ground truth for the suggestion sensor between reward events. Not a primary sensor — it doesn't directly influence action selection.

### Model

Binary sensor with learned TPR/FPR (same structure as BinarySensor).

**Question:** Compare before/after observation text — "Did the player make narrative progress?"

### Ground Truth for Itself

When game reward > 0, we have certain ground truth. Query the progress sensor and update:
- Progress said YES + reward > 0 → true positive (tp_alpha += 1)
- Progress said NO + reward > 0 → false negative (tp_beta += 1)

Sparse but certain calibration.

---

## Ground Truth Hierarchy

```
1. Game reward > 0 (certain, frequent in benchmark games)
   → Primary ground truth for ALL sensors
   → Binary sensor: update TPR/FPR
   → Categorical sensor: suggested == taken action? → correct/incorrect
   → Progress sensor: calibrate TPR/FPR

2. Progress sensor (noisy, available every turn)
   → Supplementary ground truth for categorical sensor
   → Only used when reward == 0 and suggestion was tracked
```

**Key rule:** Episode-level failure (total score = 0) says nothing about individual actions. Each action is evaluated independently by reward (when available) or progress sensor (otherwise).

---

## Benchmark Games

GLoW paper benchmark games with frequent rewards (unlike 9:05 which has only 1 reward at the end).

| Category | Game | Max Score | WT Steps | Reward Events | 1st Reward |
|----------|------|-----------|----------|---------------|------------|
| Possible | pentari | 70 | 49 | 8 | step 4 |
| Possible | detective | 360 | 51 | 26 | step 1 |
| Possible | temple | 35 | 181 | 9 | step 10 |
| Possible | ztuu | 100 | 84 | 18 | step 7 |
| Difficult | zork1 | 350 | 396 | 43 | step 4 |
| Difficult | zork3 | 7 | 273 | 7 | step 23 |
| Difficult | deephome | 300 | 327 | 56 | step 3 |
| Difficult | ludicorp | 150 | 364 | 92 | step 2 |
| Extreme | enchanter | 400 | 265 | 18 | step 24 |

With frequent rewards, the existing `reward > 0` ground truth fires regularly, making sensor learning viable without depending solely on the progress evaluator.
