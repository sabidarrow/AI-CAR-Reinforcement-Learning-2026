# Initial measured benchmark

Tested with Python 3.12.14 on Windows. The included demo checkpoint was trained for 1,000 episodes with training seed 42 and V2X enabled. The evaluation uses 50 held-out episode seeds (100042–100091) per condition with exploration and learning disabled.

| Policy | V2X | Mean distance / episode | Collision rate | Collisions / km |
| --- | --- | ---: | ---: | ---: |
| Random | Off | 97.41 m | 100% | 10.266 |
| Rule-based | Off | 1,240.42 m | 20% | 0.161 |
| Learned | Off | 835.03 m | 74% | 0.886 |
| Random | On | 97.41 m | 100% | 10.266 |
| Rule-based | On | 1,240.42 m | 20% | 0.161 |
| Learned | On | 832.50 m | 76% | 0.913 |

Mean training return improved from -111.1 over the first 100 episodes to +22.8 over the final 100. These training figures include exploration; the held-out figures above use a frozen greedy policy.

The learner improves substantially over random driving but is not as safe as the rule-based baseline. This experiment **does not demonstrate a benefit from V2X**. Both policies should be trained separately, with equal budgets and multiple random seeds, before drawing conclusions about communication. The current coarse observation bins may limit how much additional distant information helps. Background policies use only near hazards, so their V2X comparison is identical here.

The demo is a learning checkpoint, not an optimized or safe driving policy. Collisions remain frequent. Raw results and training logs are in `examples/`.

```sh
python simulate.py --watch --model examples/demo_policy.json
python simulate.py --evaluate --episodes 50 --model examples/demo_policy.json --output runs/reproduction
```

Automated validation: 10 tests cover seeded reproducibility, ring-seam collisions, swept lane-change collisions, pedestrian collisions, relay visibility, terminal learning, reset persistence, checkpoint round trips, frozen action selection, and episode limits. The browser dashboard was also exercised for rendering, pause, and model saving.
