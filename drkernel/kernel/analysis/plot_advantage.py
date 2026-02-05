import numpy as np
import math

def draw_reward_if_evaluable(rng, n):
    """
    reward distribution:
      40% -> 0
      40% -> 0.5
      20% -> U[0.5, 1.5]
    """
    u = rng.random(n)
    r = np.zeros(n, dtype=float)
    m1 = (u >= 0.4) & (u < 0.8)
    m2 = (u >= 0.8)
    r[m1] = 0.5
    if m2.any():
        r[m2] = rng.uniform(0.5, 1.5, size=int(m2.sum()))
    return r

def run_sim(
    method="grpo",
    nseeds=40,
    steps=1200,
    K=16,
    T=3,
    term_prob=0.1,
    gamma=1.0,
    lr=0.35,
    theta0=-2.0,
):
    """
    Toy 3-turn refinement.
    - Each turn can terminate with prob=term_prob (after turn1 and turn2).
    - Policy parameter theta controls evaluable/correct probability p=sigmoid(theta).
    - If evaluable: reward ~ mixture(0, 0.5, U[0.5,1.5]) with weights (0.4,0.4,0.2).
      Else: reward = 0.
    - Return-to-go G_{t} and turn-level GRPO/TRLOO baselines are computed over valid rollouts.
    """
    rews = np.zeros(steps, dtype=float)
    Nt = np.zeros(T, dtype=float)

    for s in range(nseeds):
        rng = np.random.default_rng(10000 + s)
        theta = float(theta0)

        for step in range(steps):
            p = 1.0 / (1.0 + math.exp(-theta))  # sigmoid

            alive = np.ones(K, dtype=bool)
            valid = np.zeros((K, T), dtype=bool)
            C = np.zeros((K, T), dtype=np.int8)     # evaluable indicator
            R = np.zeros((K, T), dtype=float)       # per-turn reward

            # rollout
            for t in range(T):
                valid[:, t] = alive

                Ct = (rng.random(K) < p) & alive
                C[:, t] = Ct.astype(np.int8)

                # rewards for evaluable samples at this turn
                if Ct.any():
                    R[Ct, t] = draw_reward_if_evaluable(rng, int(Ct.sum()))

                # termination after turn1 / turn2
                if t < T - 1:
                    term = (rng.random(K) < term_prob) & alive
                    alive &= ~term

            # reward-to-go return G (backward)
            G = np.zeros((K, T), dtype=float)
            G[:, T - 1] = R[:, T - 1] * valid[:, T - 1]
            for t in range(T - 2, -1, -1):
                G[:, t] = R[:, t] * valid[:, t] + gamma * G[:, t + 1]

            # advantages
            A = np.zeros((K, T), dtype=float)
            for t in range(T):
                idx = valid[:, t]
                Nt_t = int(idx.sum())
                Nt[t] += Nt_t
                if Nt_t == 0:
                    continue

                vals = G[idx, t]
                mean = float(vals.mean())

                if method == "grpo":
                    A[idx, t] = vals - mean

                elif method == "trloo":
                    if Nt_t == 1:
                        # fallback baseline=0 when leave-one-out undefined
                        A[idx, t] = vals
                    else:
                        ssum = float(vals.sum())
                        A[idx, t] = vals - (ssum - vals) / (Nt_t - 1)

                else:
                    raise ValueError("Unknown method")

            # REINFORCE score for Bernoulli evaluable indicator: d log pi(C) / d theta = C - p
            score = (C.astype(float) - p) * valid.astype(float)
            denom = float(valid.sum())
            grad = float((A * score).sum() / (denom if denom > 0 else 1.0))

            theta += lr * grad

            # metric: mean total reward per rollout (sum over 3 turns)
            rews[step] += float(R.sum(axis=1).mean())

    rews /= nseeds
    Nt_avg = Nt / (nseeds * steps)
    return rews, Nt_avg

def first_cross(arr, thresh):
    idx = np.where(arr >= thresh)[0]
    return int(idx[0]) if len(idx) > 0 else None

if __name__ == "__main__":
    gr, Nt = run_sim(method="grpo")
    tr, _  = run_sim(method="trloo")

    print("Avg N per turn:", Nt)  # should be close to [16, 14.4, 12.96]
    print("Final reward:", gr[-1], tr[-1])
    for th in [0.5, 0.8, 1.0, 1.05]:
        print("Steps to", th, ":", first_cross(gr, th), first_cross(tr, th))
