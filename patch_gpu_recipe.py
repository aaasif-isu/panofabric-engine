import sys

with open('panoengine/decentralized/decoupled_heloco/gpu_recipe.py', 'r') as f:
    text = f.read()

old_code = """    if learner_id is not None:
        cfg.dataloader = IslandDataLoaderConfig(cfg.dataloader, options.islands, learner_id)
    return cfg"""

new_code = """    if learner_id is not None:
        cfg.dataloader = IslandDataLoaderConfig(cfg.dataloader, options.islands, learner_id)
    cfg.model_spec.model.update_from_config(config=cfg)
    return cfg"""

if old_code in text:
    text = text.replace(old_code, new_code)
    with open('panoengine/decentralized/decoupled_heloco/gpu_recipe.py', 'w') as f:
        f.write(text)
    print("Patched build_recipe!")
else:
    print("Could not find code!")
