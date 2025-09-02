# benchmark1
Inputs: time grid {t_i}_{i=0..N}, cores k=1..K, schedule Sched, 
        single-step solver s_theta, vector field f_theta

Initialize:
  For all k: x^k <- x0, x_prev^k <- x0

For i = 0..N-1:
  # 并行区
  For all cores k in parallel:
    (cur, next) <- Sched(N, i, k)
    prev <- cur - 1
    Delta_i^k <- s_theta(x_cur^k, t_cur, t_next)

    if Communicate(k, i):
      r_i^k <- (x_cur^{k-1} - x_prev^k)
               + (t_next - t_prev) * ( f_theta(x_cur^{k-1}, t_cur) - f_theta(x_prev^k, t_cur) )
      x_next^k <- x_cur^k + Delta_i^k + r_i^k
      x_prev^k <- x_next^k
    else:
      x_next^k <- x_cur^k + Delta_i^k
      x_prev^k <- x_next^k

  # 步末同步（屏障）
  synchronize across k

# 可在 next == N 时按需 on-the-fly 输出各核的 x^k
