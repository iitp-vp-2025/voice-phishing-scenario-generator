# KEV server setup

KEV is a separate decision-model service. It should not be installed into the
web app's Python environment.

## One-time installation on the model server

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p "$HOME/services"
git clone --depth 1 https://github.com/jaredpalmer/kev.git "$HOME/services/kev"
cd "$HOME/services/kev"
"$HOME/.local/bin/uv" sync --extra serve --no-dev
```

`uv` installs the Python version required by KEV inside the project and leaves
the server's system Python unchanged.

## Start on CPU

CPU mode is useful for integration testing or when no GPU has enough free
memory. Kev-0.8B takes a few seconds per packed request on this server.

```bash
tmux kill-session -t kev 2>/dev/null || true
tmux new-session -d -s kev \
  "cd $HOME/services/kev && env CUDA_VISIBLE_DEVICES= KEV_CUDA_GRAPHS=0 KEV_FUSED=0 .venv/bin/python -m kev.serve --run jaredpalmer/kev-0.8b --host 127.0.0.1 --port 8009"
```

## Start on a TITAN RTX GPU

Check memory and health first. A GPU reporting `GPU requires reset` cannot be
used until an administrator resets it. Prefer a GPU UUID over a numeric index
when NVML and CUDA enumerate devices differently.

```bash
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv
```

Copy a healthy GPU UUID into `KEV_GPU_UUID`:

```bash
KEV_GPU_UUID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
tmux kill-session -t kev 2>/dev/null || true
tmux new-session -d -s kev \
  "cd $HOME/services/kev && env CUDA_VISIBLE_DEVICES=$KEV_GPU_UUID KEV_DTYPE=fp16 KEV_CUDA_GRAPHS=0 KEV_FUSED=0 .venv/bin/python -m kev.serve --run jaredpalmer/kev-0.8b --host 127.0.0.1 --port 8009"
```

TITAN RTX is a Turing GPU, so this setup uses fp16 rather than KEV's default
bf16 GPU serving mode.

## Status and logs

```bash
tmux capture-pane -pt kev -S -120
curl http://127.0.0.1:8009/v1/models
```

Stop the service with:

```bash
tmux kill-session -t kev
```

## Connect a local dashboard through SSH

Keep the KEV server bound to loopback and create an SSH tunnel instead of
opening the unauthenticated API to the network:

```bash
ssh -N -L 8009:127.0.0.1:8009 -p <SSH_PORT> <USER>@<SERVER>
```

The local app then uses:

```dotenv
KEV_API_URL=http://127.0.0.1:8009
KEV_MODEL=kev-latest
```
