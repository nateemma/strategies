"""Stage 3 config: window length (32, 64) on the label variant the screen rule selects.
Pick the PROMOTE variant with the largest RIDGE pooled MCC across screen_20/21; else the baseline label."""
import json, yaml, sys
best=None
for p in (20,21):
    try: d=json.load(open(f"research/clfstudy/report/screen_{p}.json"))
    except FileNotFoundError: continue
    for v in (d.get("variants") or d if isinstance(d,list) else d.get("variants",[])):
        if v.get("verdict")=="PROMOTE" and v.get("input_set","ind24")=="ind24":
            rp=v.get("ridge_pooled")
            if rp is not None and (best is None or rp>best[0]): best=(rp, v.get("label"))
label = best[1] if best and isinstance(best[1],dict) else {"name":"triple_barrier","horizon":48,"up":0.02,"down":0.02}
print("stage-3 base label:", label, "(from PROMOTE variant)" if best else "(baseline; no PROMOTE)")
T={"lr":0.0003,"stride":2}; TR={"lr":0.0003,"stride":2,"weight_decay":0.01}
cfgs=[]
for L in (32,64):
    cfgs.append({"arch":"RIDGE","hp":{},"input_set":"ind24","seq_len":L,"split_set":"v1","train":TR,"label":label,"phase":22,"seeds":[0,1,2],"tags":["screen",f"seq{L}"]})
    cfgs.append({"arch":"MLP_FAM","hp":{"width":64},"input_set":"ind24","seq_len":L,"split_set":"v1","train":T,"label":label,"phase":22,"seeds":[0,1],"tags":["screen",f"seq{L}"]})
yaml.safe_dump({"phase":22,"description":"019 stage 3: window length 32/64 on the selected label variant","gate_from":21,"configs":cfgs}, open("research/clfstudy/configs/phase_22.yaml","w"), sort_keys=False)
