# Inactive published Physics meanSAGE context

This is a source-derived Torch port of the pinned GNNBenchmark TensorFlow author meanSAGE at commit1e72912a0810cdf27ae54fd589a3b43358a2b161. It is not a bitwise TensorFlow replication. The OGB author SAGE model is not transplanted. Root admits and executes source/runtime/provider/resource/protocol custody. No numerical run or current predictive outcome is supplied here.

## Exact adopted author recipe

Two meanSAGE layers8415→32→5, dropout.4 before each. Sparse raw binary input is not row-normalized. Pinned util.py drops stored sparse values, preserves indices even for dropped zeros, and dense hidden values receive elementwise dropout. The same dropped input feeds the aggregate and skip matrices. Each layer forms z=mean_A(Drop(h)Wagg)+Drop(h)Wskip+b, then divides each row by max(L2norm(z),1). ReLU only follows the first layer; final logits are also clipped this way. All matrix initializers have Glorot uniform law, all biases start at zero. Parameter count is538917.

The already-prepared530417-entry complete common support includes34493 self edges; no consumer adds self loops or symmetry. TRAIN mean CE plus .05sum(Wagg0²+Wskip0²) is minimized. Only first-layer matrices are regularized, reflecting TFslim.1 times half squared norm. TF1 Adam LR.001/beta.9,.999/epsilon1e-8 is explicitly ported: lr_t=lr*sqrt(1-beta2_power)/(1-beta1_power), update p-=lr_t*m/(sqrt(v)+eps); float32 beta powers are checkpoint state. This avoids silent Torch-default epsilon rescaling. Torch kernels, RNG mapping and reduction order differ from TensorFlow; source equations/distributions are ported, not bit identity.

SELECT total CE+L2 is evaluated after every update. Actual author KDDCriterion uses <=, selecting the latest tied minimum and resetting patience on ties. Stop at50 consecutive strictly worse SELECT totals or100000 updates; no grid or retry. The port preserves a full selected model, Adam and Python/NumPy/CPU/CUDA RNG snapshot and restores it before export, stronger state custody than the author's trainable-variable-only final restore.

Published Physics GS-mean93.0±.8% from Table1 of arXiv1811.05868 is competence context. The author averaged100splits and20initializations per split. Root uses ten different prospective custom stratified paired masks:20TRAIN/30SELECT/30CAL per class, rest CONFIRM (100/150/150/34093). Stratification uses custodian truth; these are not official masks. Reserving150CAL nodes changes the final test role. No exact reported-score reproduction or SOTA claim follows from this packet.

## Callable integration

Call `consumer.run(cfg,output,root_validate)` once for all ten contexts. cfg controls:
- execution_enabled=true, device="cuda:0", seeds=[8101,8219,8363,8501,8663,8819,9001,9199,9403,9623];
- point equals the complete `author_model.POINT` dictionary;
- limits exact{wall_seconds,cuda_bytes,rss_bytes}, each finite positive integer, plus root external watchdog;
- profile includes dataset="CoauthorPhysics",nodes34493,input_dim8415,classes5,edges530417,self_edges34493,train_rows100,select_rows150,cal_rows150,confirm_rows34093;
- role_blocks all ten seed strings, each exact{train,select,cal_ids,confirm_ids}: path strings. No CAL/CONFIRM truth path is accepted.

The callback `root_validate(cfg,action="fixed_Physics_author_meanSAGE_context_fits",source_receipts={...})` must return primitive JSON custody with status="ROOT_AUTHOR_CONTEXT_FITS_ADMITTED". It binds actual host/GPU/runtime, all source and all ten role NPZ/ID hashes, identical full public X/support in all TRAIN NPZs, provider/prospective masks, resource limits/qualification and source authority. Admission precedes numerical imports/data reads. The consumer loads full X/support from the first safe TRAIN once; later blocks read tiny ids/y only. Safe TRAIN fields exact{x,edge_index,ids,y}; SELECT exact{ids,y}. CAL/CONFIRM files contain IDs only.

Each `author_meanSAGE_M1_seed<seed>/` contains trace.jsonl, selected.pt, RESULT.json and selected_SELECT/CAL/CONFIRM.npz. Checkpoint fields include model,optimizer,global_rng,seed,selected_step,SELECT_total_loss/CE/L2/correct/rows,custody,config_sha256 and state fingerprints. Archive fields exact{ids,member_log_probs,native_member_argmax,native_pool_argmax}: int64[rows],float32[1,rows,5],int64[1,rows],int64[rows]. Native decision authority is retained before the common float64 temperature readout. No y or CAL/CONFIRM metrics are exported.

`COMPLETE_AUTHOR_CONTEXT.json` returns/writes ten complete records. Each has block/seed,arm="author_meanSAGE_M1",status="SELECTED_COMPLETE",members=1,fits=[{status,selected_state_receipt,costs,...}],selected_CAL_archive and selected_CONFIRM_archive receipts. Combine these10 banks/10 fits with the core50banks/80fits only after full closure. Root/readout owner handles all60 fixed CAL temperatures and one scorer-only complete CONFIRM release. A source seal grants no fit/calibration/confirmation release.
