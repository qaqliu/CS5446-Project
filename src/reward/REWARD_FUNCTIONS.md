# Reward functions in the five-recipe comparison

This guide covers the five training rewards compared in the reward experiments: Champion, anticipatory right of way, congestion aware journey time, unlocking other trains, and deadline slack rescue. The other four rewards each extend Champion. The formulas below use the values in their `config/protocol/shared_ppo_random_maps_*.yaml` recipes.

These are **training rewards**, calculated for each train after an environment step. Evaluation's *completion rate* and *competition score* are separate metrics; a training reward of `+10` is not a competition score of `10`.

## Common notation

For train $i$ at step $t$:

- $d_i(t)$: direction-aware shortest remaining distance along legal rails to its target. Before departure, the origin is used; once the train is done, distance is `0`. An unreachable distance is represented as `1000`.
- $D_i(t)$: `1` only on the first step of the episode when the dependency-based deadlock checker flags train $i$; otherwise `0`.
- $A_i(t)$: `1` only on the first step when train $i$ reaches `DONE`; otherwise `0`.
- $I_{\mathrm{name}}$: a yes/no indicator. It equals 1 when the named condition is true and 0 otherwise; each condition is defined below.
- $\operatorname{clip}(x,a,b)=\min(b,\max(a,x))$. In particular, $\operatorname{clip}(x,0,1)$ is zero below zero and one above one.

The shared Champion term is

$$
C_i(t)=0.01\,[d_i(t-1)-d_i(t)]-5D_i(t)+10A_i(t).
$$

The deadlock penalty and arrival bonus can each be paid at most once per train per episode. The distance term can be positive for progress, negative for movement away from the target, or zero for no distance change. The implementation returns this shaped reward in place of Flatland's default reward. See [rewards.py](rewards.py) and the [Champion recipe](../../config/protocol/shared_ppo_random_maps_champion_reward.yaml).

## Champion reward

**Key characteristic:** A simple combination of rail-distance progress, successful arrival, and deadlock avoidance.

**Formula:** $r_i^{\mathrm{champion}}(t)=C_i(t)$.

**Terms:** `0.01 × distance decrease` gives frequent feedback before arrival; `−5` discourages a train from becoming deadlocked; `+10` strongly rewards reaching the destination. There is no separate per-step time cost in this recipe; PPO's discounting can still make earlier rewards more valuable during learning.

### Limitations

- Shortest rail distance ignores other trains and congestion. A move that looks good for one train may block another train.
- The deadlock penalty arrives only after the checker detects a deadlock, and each train receives it only once. This reward has no explicit per-step time cost or deadline signal.

### Short summary

The train earns a little for getting closer to its destination and a big bonus for arriving. It loses points the first time it gets stuck in a detected deadlock.

## Anticipatory right of way

**Key characteristic:** Reward movement or yielding that reduces a *predicted* near-term conflict on shared track.

**Formula:**

$$
r_i^{\mathrm{right\ of\ way}}
=C_i-0.01I_{\mathrm{unfinished}}
+0.10I_{\mathrm{moved}}\operatorname{clip}(R_{\mathrm{prev}}-R_{\mathrm{now}},-2,2)
+0.05I_{\mathrm{clear}}
+0.05I_{\mathrm{yield}}\min(q_i,2).
$$

**Terms and conditions:**

- **Indicator terms:** $I_{\mathrm{unfinished}}=1$ while this train is not DONE; $I_{\mathrm{moved}}=1$ only if it is unfinished and its movement marker changed. $I_{\mathrm{clear}}=1$ only if it is unfinished and meets the track-clearance conditions below. $I_{\mathrm{yield}}=1$ only if it is unfinished, its marker is unchanged, and it is ready to depart or on the map. Each is zero otherwise.

- **Forecast risk $R_i$:** Each unfinished train is projected along a greedy, distance-decreasing rail path up to three steps ahead. Two trains are considered conflicting when their projected visits to a cell are at most one step apart. A train pair's risk is the largest $1/(1+\max(a,b))$ over such shared cells, where $a,b$ are their projected visit steps; $R_i$ sums that risk across peers. Nearer conflicts count more. A ready-to-depart train's origin is projected at step one.
- **Time cost `−0.01`:** Applied on every step while the train is unfinished, including a step spent waiting.
- **Risk change:** Paid only when the train's movement marker (rail configuration or within-cell speed progress) changes. A fall in risk earns a positive amount; a rise gives a negative amount. Clipping limits this component to the range −0.20 to +0.20 per step.
- **Track clearance `+0.05`:** Paid when a moving train leaves a previously contested cell, enters a cell that is no longer contested, and decreases its rail distance to the target.
- **Useful yield +0.05 per peer:** The train is eligible only when its movement marker is unchanged and it is ready to depart or on the map. $q_i$ counts previously conflicting peers that moved and no longer conflict with it, excluding peers already credited to this train during the episode; at most two peers are paid in one step. Waiting without a peer advancing earns no yield bonus.

All four added terms are skipped when the train is DONE, including on the arrival step. Champion's one-time arrival bonus still applies on that step.

This is a short-horizon heuristic, not a right-of-way rule that guarantees safe passage. It forecasts a greedy shortest route rather than the other policies' actual future actions, so it can miss later conflicts or misjudge a detour. See [traffic_rewards.py](traffic_rewards.py) and the [right-of-way recipe](../../config/protocol/shared_ppo_random_maps_anticipatory_right_of_way.yaml).

### Limitations

- The forecast follows one greedy shortest route for at most three steps. It can miss later conflicts, alternative routes, or a train that chooses a different action.
- A predicted risk reduction or released peer does not prove that the train caused a safe outcome. The bonus can favor a locally helpful move that creates a problem farther ahead.

### Short summary

The trains look a few moves ahead for places where their routes might clash. A train can earn points by clearing that track or by waiting when the wait actually lets another train through.

## Congestion aware journey time

**Key characteristic:** Reward a move that reduces estimated travel time after accounting for nearby occupied track and projected traffic.

**Formula:**

$$
r_i^{\mathrm{congestion}}
=C_i-0.01I_{\mathrm{unfinished}}
+0.05I_{\mathrm{changed}}\operatorname{clip}(\mathrm{ETA}_{\mathrm{prev}}-\mathrm{ETA}_{\mathrm{now}},-3,3).
$$

$I_{\mathrm{unfinished}}=1$ while the train is not DONE. $I_{\mathrm{changed}}=1$ when its movement marker changes; reaching DONE can change that marker, so the ETA term can apply on the arrival step. Both indicators are zero when their conditions are false.

The ETA is calculated by a recursive search. Let $x$ be a rail configuration, $h$ the number of searched successors, $S$ the configurations already visited, and $\tau_i=1/v_i$ when speed $v_i>0$ (otherwise $\tau_i=1$). Let $\mathcal{N}_i(x,S)$ contain legal, target-reachable successors not in $S$. Then the search is

$$
\begin{aligned}
E_i(x,h,S)&=d_i(x)\tau_i
&&\text{if }d_i(x)\leq0\text{ or }h\geq3,\\
E_i(x,h,S)&=
\min_{y\in\mathcal{N}_i(x,S)}
\left[
\tau_i+
\frac{2\,O(y)+\min(F_i(y,h+1),2)}{h+1}
+E_i(y,h+1,S\cup\{y\})
\right]
&&\text{if }\mathcal{N}_i(x,S)\neq\varnothing,\\
E_i(x,h,S)&=d_i(x)\tau_i
&&\text{otherwise.}
\end{aligned}
$$

The train's current configuration (or origin before departure) starts the search at $h=0$. The resulting $\mathrm{ETA}_i$ is $E_i$ at that starting point, except that a finished train has ETA zero and an unreachable starting point has ETA 1000. $O(y)=1$ if another train currently occupies successor cell $y$, otherwise zero; $F_i(y,h+1)$ counts other trains projected to use that cell within one step of projected visit $h+1$. The denominator $h+1$ discounts congestion farther ahead. At the target, after three successors, or when no legal unvisited successor is available, the remaining shortest-distance time is used.

**Terms and conditions:** The −0.01 time cost applies while unfinished. The ETA-change term is paid when the movement marker changes, including a change caused by arrival: reducing ETA helps, increasing ETA hurts, and clipping limits its contribution to −0.15 through +0.15 per step. A longer route may look better than a shorter but occupied route, though the forecast is only three steps deep. See [traffic_rewards.py](traffic_rewards.py) and the [congestion recipe](../../config/protocol/shared_ppo_random_maps_congestion_aware_journey_time.yaml).

### Limitations

- Occupancy and projected demand are snapshots, so the ETA can be wrong when other trains change direction or stop. The search sees only three successor cells before falling back to shortest distance.
- The reward measures each train's estimated travel time. It may prefer a locally fast move that increases congestion for the group.

### Short summary

The train is rewarded for choosing a route that looks faster after nearby traffic is considered. A slightly longer free route can look better than a short route blocked by other trains.

## Unlocking other trains

**Key characteristic:** Give a train credit when leaving a cell releases another train that had no free legal exit.

**Formula:**

$$
r_i^{\mathrm{unlock}}
=C_i-0.02I_{\mathrm{actionable}}
+\min\!\left(1,\sum_{j\in U_i}0.5\,(0.5)^{h_{ij}-1}\right).
$$

**Terms and conditions:**

- **Actionable cost:** $I_{\mathrm{actionable}}=1$ while the train is ready to depart, moving, or stopped on the map, and zero otherwise. The cost is −0.02 in those states; finished trains do not pay it.
- **Direct unlock:** A peer is blocked if *all* its legal next cells were occupied, or its origin was occupied before departure. When train $i$ vacates a cell needed by that peer and leaves it free (or lets that peer enter), the peer can enter set $U_i(t)$ with queue depth $h_{ij}=1$, worth `+0.5`.
- **Queue credit:** If the newly unblocked peer had been blocking another peer, the original blocker can receive further credit along that dependency chain: `+0.25` at depth two and `+0.125` at depth three. The search stops after depth three, total credit is capped at `+1` per step, and each blocker–beneficiary pair can be credited only once per episode.

The reward uses actual movement of a former blocker and the before/after occupancy, rather than merely guessing that someone might be freed. It still measures a local queue release, not whether the whole network will eventually clear. See [cooperative_rewards.py](cooperative_rewards.py) and the [unlocking recipe](../../config/protocol/shared_ppo_random_maps_unlocking_other_trains.yaml).

### Limitations

- The bonus requires a peer to have been completely blocked. Relieving partial congestion or preventing a future blockage earns no unlock credit.
- Queue credit stops after three links and each blocker-beneficiary pair is credited once. A local release can still lead to another blockage later.

### Short summary

When a train moves out of the way and another train can finally go, the first train gets points for helping. It may also get smaller points for helping trains farther back in the queue.

## Deadline slack rescue

**Key characteristic:** Give extra progress credit to trains at risk of missing their latest arrival time, while penalizing further loss of time margin.

**Formula:**

$$
\begin{aligned}
r_i^{\mathrm{deadline}}
={}&C_i-0.02I_{\mathrm{actionable}}I_{\mathrm{clock}}\\
&+I_{\mathrm{previously\ actionable}}\rho_{\mathrm{prev}}
\left[
0.2\operatorname{clip}(\eta_{\mathrm{prev}}-\eta_{\mathrm{now}},0,1)
-0.1\operatorname{clip}(s_{\mathrm{prev}}-s_{\mathrm{now}},0,1)
\right].
\end{aligned}
$$

where

$$
\eta_i(t)=
\begin{cases}
0,&\text{if the train is DONE},\\
1000,&\text{if }v_i\leq0,\\
\max\!\left(0,\dfrac{d_i(t)-f_i(t)}{v_i}\right),&\text{otherwise},
\end{cases}
\qquad
s_i(t)=L_i-e_t-\eta_i(t),\qquad
\rho_i(t-1)=\operatorname{clip}\!\left(\frac{10-s_i(t-1)}{10},0,1\right).
$$

**Indicator terms:** $I_{\mathrm{actionable}}=1$ if the current state is ready to depart, moving, or stopped. $I_{\mathrm{previously\ actionable}}=1$ if the previous state was one of those three. $I_{\mathrm{clock}}=1$ if the environment elapsed-step count increased. Each is zero otherwise.

**Terms and conditions:** $\eta_i$ estimates remaining travel time from rail distance $d_i$, fractional progress $f_i$ within the current cell, and maximum speed $v_i$; $f_i=0$ before departure. $L_i$ is the train's latest allowed arrival time and $e_t$ is the environment's elapsed step count, so slack $s_i$ is the time margin after its estimated travel time. The urgency weight $\rho_i$ is zero when prior slack is at least ten steps, rises as slack falls, and reaches one at zero or negative slack.

The `+0.2` term rewards an ETA reduction of at most one unit per step; the `−0.1` term penalizes losing up to one unit of slack. Both are multiplied by *previous-step* urgency and apply only if the train was previously ready, moving, or stopped. The `−0.02` cost applies when it is currently in one of those states and the environment clock advances. See [cooperative_rewards.py](cooperative_rewards.py) and the [deadline recipe](../../config/protocol/shared_ppo_random_maps_deadline_slack_rescue.yaml).

### Limitations

- Its ETA uses shortest rail distance and speed but does not account for other trains or congestion. An optimistic ETA can hide an approaching deadline problem.
- With at least ten steps of estimated slack, the urgency term is zero. Progress that helps other trains or avoids a future conflict is not directly measured by this added term.

### Short summary

A train that is close to being late gets extra points for making useful progress. Losing more time when it is already at risk costs it points.
