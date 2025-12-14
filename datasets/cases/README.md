# Dataset Overview

This benchmark includes a variety of PDE-ODE coupled datasets across different geometries and physics. Below is the summary of currently available datasets:

| Dataset        | Coor. system | Data struct.   | Phys. parameters                    | n_step | n_traj |
|----------------|--------------|----------------|-------------------------------------|--------|--------|
| IgnitHIT       | (x, y)       | 128×128        | T, rho, Ux, Uy, 8Yis                | 30     | 30     |
| EvolveJet      | (x, y)       | 256×256        | T, rho, Ux, Uy, 36Yis               | 40     | 40     |
| SupCavityFlame | (x, r)       | 1×3M           | T, p, rho, Ux, Uy, 9Yis             | 100    | 9      |
| PlanarDet      | (x, y)       | 832x384        | T, p, pMax, rho, Ux, Uy, 9Yis       | 50     | 9      |
| ObstacleDet    | (x, r)       | 1x9M           | T, p, rho, Ux, Uy, 2Yis             | 50     | 6      |
| SymmCoaxFlame  | (x, r)       | 1x0.3M         | T, p, rho, Ux, Uy, 17Yis            | 36     | 12     |
| PropHIT        | (x, y, z)    | 1024×128×128   | T, rho, Ux, Uy, 9Yis                | 35     | 7      |
| ReactTGV       | (x, y, z)    | 256×256×256    | T, p, rho, Ux, Uy, 9Yis             | 20     | 21     |
| PoolFire       | (x, y, z)    | 80×80×200      | T, p, rho, Ux, Uy, 5Yis             | 21     | 15     |
| MultiCoaxFlame | (x, r)       | 1x13.4M        | T, p, rho, Ux, Uy, 17Yis            | 91     | 5      |
| FacadeFire     | (x, r)       | 1x2.52M        | T, p, rho, Ux, Uy, 5Yis             | 100    | 9      |
