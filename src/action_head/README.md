# VLA-JEPA — Action Head con DDP e batch bilanciati per task

Il pacchetto va in `workspace/src/action_head/`, accanto alla directory **già esistente** `src/vla_jepa_offline/`. Copiare i cinque `.sbatch` in `workspace/slurm/`. Non sovrascrivere il codice E3/E4.

## Architettura e dati

- **Input:** due immagini RGB 224×224 (frontale e gripper) e istruzione testuale. Nessuna propriocezione per la policy.
- **Target:** chunk di 7 azioni 7D, già scalate per `1/0.05` nella costruzione dei PKL. L'ultima action nulla reale rimane valida; il padding è mascherato.
- **Model:** checkpoint E1–E4; Qwen e relative embedding congelati, World Model disabilitato, action head DiT addestrata con il forward originale di LeRobot.
- **Loss:** flow matching `reduction="none"`, ossia media sui timestep reali di ogni osservazione e poi media sulle osservazioni del batch globale.
- **Normalizzazione:** soltanto `ACTION=MIN_MAX`. Immagini convertite in `float [0,1]`, senza ulteriori crop o resize. Output: clipping `[-1,1]` prima della MIN_MAX inversa, senza pre-snap né binarizzazione del gripper.
- **Controller:** riceve i sette valori nelle unità del dataset, poi applica `×0.05` alle sei componenti cinematiche. Nessuna modifica funzionale al controller.

## Un'epoca e un optimizer step

Il sampler costruisce un ordine completo di tutti i timestep per ogni task. Durante ogni epoca **ogni frame originale viene utilizzato esattamente una volta**; i task più corti ricevono frame aggiuntivi estratti casualmente per raggiungere il numero di frame del task più lungo.

I frame sono intercalati fra traiettorie dello stesso task. L'ordine dei dodici task cambia a ogni riga, per evitare rank legati a task fissi. Ogni riga di 12 campioni contiene **un'osservazione per ognuno dei 12 task**.

- `4 GPU × batch_size 3 = 12` osservazioni per microbatch globale.
- `grad_accum=2`: normalmente 24 osservazioni per optimizer step, **2 per task**.
- Se il task più lungo ha un numero dispari di frame, l'ultimo optimizer step ha 12 osservazioni, **1 per task**, senza drop/duplicazioni artificiali.
- Passi per epoca = `ceil(max_frame_task / grad_accum)`.
- Seed iniziale **identico per tutti i rank**, senza somma del rank. Lo shuffle condiviso è suddiviso per rank dalla posizione degli indici, non dal seed.

## Split e validazione

- Split a livello di traiettoria: per ciascuno dei 12 task **1 TFRecord + 1 scripted in validazione**; tutti gli altri episodi nel training.
- Manifest condiviso da E1–E4. Statistiche MIN_MAX calcolate sul solo train e salvate automaticamente.
- Validazione senza oversampling: tutti i frame delle 24 traiettorie, con **macro-average della loss sui 12 task** e media pesata per frame. Il rumore della validazione è riproducibile e non consuma lo stato casuale del training.

## Installazione e comandi

Modificare i quattro percorsi `pretrained_paths` nel file `src/action_head/configs/action_head.yaml` e verificare dataset e root di output. Dalla radice del workspace:

```bash
mkdir -p logs
export TRAINING_ROOT=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training
export PYTHONPATH="$PWD/src:$PWD:$TRAINING_ROOT:$TRAINING_ROOT/multi_task_il/datasets:${PYTHONPATH:-}"

# Solo CPU: genera split e statistiche (non carica Qwen).
python -m action_head.train --config src/action_head/configs/action_head.yaml --stats-only

# Solo CPU: verifica l'identità delle action head iniziali E1-E4.
python -m action_head.train --config src/action_head/configs/action_head.yaml --check-heads

# Smoke DDP 4 GPU su E1 (output separato).
sbatch slurm/action_head_smoke.sbatch
# Per cambiare esperimento smoke:
sbatch --export=ALL,EXPERIMENT=E3 slurm/action_head_smoke.sbatch

# Addestramenti (ognuno usa 4 GPU):
sbatch slurm/action_head_E1.sbatch
sbatch slurm/action_head_E2.sbatch
sbatch slurm/action_head_E3.sbatch
sbatch slurm/action_head_E4.sbatch
```

Gli script Slurm riprendono `gpuq`, account, QoS, nodi esclusi, ambiente Conda e PYTHONPATH del training offline E3/E4.

## Checkpoint e resume

Ogni esperimento esporta in `checkpoints/action_head/E1/`, ecc., due slot alternati `checkpoint_A` e `checkpoint_B`. Ciascuno contiene `pretrained_model/` (pesi LeRobot e processor serializzati), `trainer_state.pt` (optimizer, scheduler e RNG) e `metadata.json`. Per lo smoke usare `checkpoints/action_head/smoke/E1/`, ecc.

La directory `pretrained_model/` include anche **`policy_postprocessor_ur5e.json`**, nome richiesto dal runtime ROS esistente. Il resume avviene dall'inizio dell'epoca successiva all'ultima salvata. Rilanciare lo stesso script per riprendere.

## Note di validazione

1. **Prima** dei quattro training eseguire `--check-heads`. Il codice non reinizializza le DiT: checkpoint 8D incompatibili non sono adattati silenziosamente.
2. Eseguire lo smoke DDP sulle A100 e caricare il checkpoint esportato nel runtime senza robot, prima dell'esperimento reale. Questi test richiedono l'ambiente LeRobot del cluster e i checkpoint E1–E4.
3. Il vecchio `test_setup.py` e `run_server.sh` possono avere controlli rigidi sui nomi dei vecchi file dei processor: adeguare i controlli di preflight se necessario, senza cambiare la logica del controller.
4. L'algoritmo di bilanciamento ripete **frame**, non intere traiettorie: tutti i frame originali contribuiscono e i duplicati cambiano casualmente a ogni epoca.

## Test CPU

```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" python -m unittest discover -s src/action_head/tests -v
```
