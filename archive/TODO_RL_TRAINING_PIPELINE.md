# TODO - RL Intrusion Detection Training Pipeline

## Step 1: Implement training loop
- [ ] Update `train.py` to run simulator + environment + agent
- [ ] Run 500–1000 episodes (default configurable)
- [ ] Track accuracy + rewards
- [ ] Print progress every 50 episodes
- [ ] Save learning curve data (CSV/JSON)
- [ ] Plot Episode vs Accuracy using matplotlib
- [ ] Ensure `python train.py` is runnable end-to-end

## Step 2: Load hyperparameters from JSON (recommended)
- [ ] Update `config.py` to read `hyperparameters.json` when available
- [ ] Ensure overrides fall back to defaults safely

## Step 3: (Optional) Reproducibility
- [ ] Implement `utils.set_seed()` to set Python random seed (and numpy if present)

## Step 4: Test
- [ ] Run `python train.py`
- [ ] Confirm logs show improving accuracy/rewards
- [ ] Confirm graph displayed
- [ ] Confirm learning curve file created

