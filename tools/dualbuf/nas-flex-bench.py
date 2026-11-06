#!/usr/bin/env python3
"""Flex-band measurement on the NAS: starts the dual-buffer Ollama image as a throw-away test container
and measures, for each model, with 0, 1 and 2 layers taken off the GPU (--flex-layers / LLAMA_FLEX_TEST):

  - generation speed and long-prompt speed (tokens/s, from the Ollama API timings)
  - freed VRAM (sysfs mem_info_vram_used of the GPU, and the "device free memory" of the llama-server log)
  - time of the take-off and of the put-back (weights and K/V, from the llama-server log)
  - number of graph splits after the move (sched_reserve log lines)
  - whether the output is sane, and whether it differs from the k=0 reference / from the GGML_CUDA_DISABLE_GRAPHS=1 run

Every case runs twice: HIP graphs as they are and with GGML_CUDA_DISABLE_GRAPHS=1.

Needs: python3 (standard library only), docker, root or docker group. The models directory is mounted read-only,
so use_mmap=false is sent as a request option (no Modelfile variant is created) and LLAMA_ARG_LOAD_MODE=none is set.

Usage on the NAS (zsh is fine, run it with python3):
  sudo python3 tools/dualbuf/nas-flex-bench.py --image ghcr.io/csaki89/ollama-dualbuf:<tag>
  sudo python3 tools/dualbuf/nas-flex-bench.py --image ... --models qwen3-vl:8b --layers 0,1 --no-graphs-off   (subset)
  python3 tools/dualbuf/nas-flex-bench.py --dry-run                                                          (print only)

Stop the production Ollama's model first (or let it idle with keep_alive 0): it shares the GPU, and its VRAM use
would distort the VRAM figures. The script warns if the GPU already holds more than --vram-idle-max MiB.
Results: <out>/results.md, <out>/results.json, and the container logs in <out>/logs/.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- helpers

def sh(cmd, check=True, timeout=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError("command failed: %s\n%s" % (cmd, r.stderr.strip()))
    return r.stdout


def vram_used_mib(path):
    try:
        with open(path) as f:
            return int(f.read().strip()) / 1048576.0
    except OSError:
        return None


def http_json(url, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


SENTENCES = [
    "The quick brown fox jumps over the lazy dog while the engineers measure the prompt processing speed.",
    "A hybrid CPU and GPU setup moves transformer layers between the two devices at runtime.",
    "Memory bandwidth, not compute, limits the token generation speed of large language models.",
    "Each layer holds attention weights, feed forward weights and a slice of the key value cache.",
]


def long_prompt(tag, words):
    out, n, i = ["Document %s. Read the text and answer with a one line summary at the end." % tag], 0, 0
    while n < words:
        s = "%d. %s" % (i, SENTENCES[i % len(SENTENCES)])
        out.append(s)
        n += len(s.split())
        i += 1
    out.append("Summary:")
    return "\n".join(out)


# ---------------------------------------------------------------- log parsing

RE_W = re.compile(r"layer (\d+) moved to (\S+) \(([\d.]+) MiB, ([\d.]+) ms: alloc ([\d.]+), copy ([\d.]+), free ([\d.]+);"
                  r".*?device free memory ([\d.]+) MiB")
RE_KV = re.compile(r"layer (\d+) K/V moved to (\S+) \(([\d.]+) MiB, (\d+) cells, ([\d.]+) MiB copied, ([\d.]+) ms")
RE_SPLIT = re.compile(r"sched_reserve: graph(?: \(pp bs=\d+, tg bs=\d+\))?: nodes = [^,]+, splits = (\d+(?: / \d+)?),")


def parse_log(text):
    """Events in log order: ('w'|'kv'|'splits', fields...)."""
    ev = []
    for line in text.splitlines():
        m = RE_W.search(line)
        if m:
            il, dst, mib, ms, al, cp, fr, free = m.groups()
            ev.append(("w", int(il), dst, float(mib), float(ms), float(free)))
            continue
        m = RE_KV.search(line)
        if m:
            il, dst, mib, cells, cmib, ms = m.groups()
            ev.append(("kv", int(il), dst, float(mib), float(ms)))
            continue
        m = RE_SPLIT.search(line)
        if m:
            ev.append(("splits", m.group(1)))
    return ev


def summarize_moves(ev):
    """take = weights moved to the CPU, put = moved back to a GPU buffer; the K/V event of a layer follows its weight event
    (its destination is the GPU host buffer when taken, so the kind comes from the weight event)."""
    res = {"take": {}, "put": {}, "splits_after_move": []}
    kind_of = {}
    last_kind = None
    for e in ev:
        if e[0] == "w":
            kind = "take" if e[2].startswith("CPU") else "put"
            kind_of[e[1]] = kind
            d = res[kind].setdefault(e[1], {"w_ms": 0.0, "kv_ms": 0.0, "w_mib": 0.0, "kv_mib": 0.0, "free_mib": None})
            d["w_ms"], d["w_mib"], d["free_mib"] = e[4], e[3], e[5]
            last_kind = kind
        elif e[0] == "kv":
            kind = kind_of.get(e[1])
            if kind:
                d = res[kind].setdefault(e[1], {"w_ms": 0.0, "kv_ms": 0.0, "w_mib": 0.0, "kv_mib": 0.0, "free_mib": None})
                d["kv_ms"], d["kv_mib"] = e[4], e[3]
        elif e[0] == "splits" and last_kind:
            res["splits_after_move"].append((last_kind, e[1]))
    res["initial_splits"] = [e[1] for e in ev if e[0] == "splits"][:2]
    return res


# ---------------------------------------------------------------- one case

class Case:
    def __init__(self, a, model, k, graphs_off):
        self.a, self.model, self.k, self.graphs_off = a, model, k, graphs_off
        self.name = "%s_k%d_%s" % (re.sub(r"[^A-Za-z0-9]+", "-", model), k, "nographs" if graphs_off else "graphs")
        self.cname = "flexbench-" + self.name.lower()

    def env(self):
        a = self.a
        e = {
            "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
            "OLLAMA_HOST": "0.0.0.0:11434",
            "OLLAMA_MODELS": "/models",
            "OLLAMA_NOPRUNE": "1",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_KEEP_ALIVE": "30m",
            "OLLAMA_CONTEXT_LENGTH": str(a.ctx),
            "OLLAMA_DEBUG": "1",
            "LLAMA_ARG_LOAD_MODE": "none",
            "LLAMA_ARG_FLEX_LAYERS": str(a.band),
            "LLAMA_FLEX_TEST_RESERVE": "1",  # re-reserve after a move: prints the graph splits
        }
        if a.dual_buffer:
            e["LLAMA_ARG_DUAL_BUFFER"] = "1"
        if self.k > 0:
            steps = ["1:take"] * self.k + ["%d:put" % a.put_at] * self.k
            e["LLAMA_FLEX_TEST"] = ",".join(steps)
        if self.graphs_off:
            e["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        return e

    def start(self):
        a = self.a
        cmd = ["docker", "run", "-d", "--name", self.cname,
               "--device", "/dev/kfd", "--device", a.render_node,
               "--group-add", "video", "--group-add", "render",
               "--memory", a.memory, "--memory-swap", a.memory, "--cpus", str(a.cpus),
               "--ulimit", "memlock=-1:-1",
               "-p", "%d:11434" % a.port,
               "-v", "%s:/models:ro" % a.models_dir]
        for k, v in self.env().items():
            cmd += ["-e", "%s=%s" % (k, v)]
        cmd.append(a.image)
        if a.dry_run:
            print(" ".join(cmd))
            return
        sh(["docker", "rm", "-f", self.cname], check=False)
        sh(cmd)
        for _ in range(120):
            try:
                http_json("http://127.0.0.1:%d/api/version" % a.port, timeout=3)
                return
            except (urllib.error.URLError, OSError, ValueError):
                time.sleep(1)
        raise RuntimeError("Ollama did not become ready, see `docker logs %s`" % self.cname)

    def logs(self):
        r = subprocess.run(["docker", "logs", self.cname], capture_output=True, text=True)
        return r.stdout + r.stderr

    def stop(self, save=True):
        if self.a.dry_run:
            return
        if save:
            os.makedirs(os.path.join(self.a.out, "logs"), exist_ok=True)
            with open(os.path.join(self.a.out, "logs", self.name + ".log"), "w") as f:
                f.write(self.logs())
        sh(["docker", "rm", "-f", self.cname], check=False)

    def gen(self, prompt, n_predict, extra=None):
        a = self.a
        opts = {"num_ctx": a.ctx, "temperature": 0, "seed": 1, "num_predict": n_predict, "use_mmap": False}
        ng = a.num_gpu.get(self.model)
        if ng is not None:
            opts["num_gpu"] = ng
        if extra:
            opts.update(extra)
        body = {"model": self.model, "prompt": prompt, "stream": False, "keep_alive": "30m", "options": opts}
        return http_json("http://127.0.0.1:%d/api/generate" % a.port, body, timeout=1800)

    @staticmethod
    def tps(r, kind):
        n, d = r.get("%s_count" % kind, 0), r.get("%s_duration" % kind, 0)
        return (n / (d / 1e9)) if d else 0.0

    def wait_moves(self, want_take, want_put, tmo):
        t0 = time.time()
        while time.time() - t0 < tmo:
            ev = parse_log(self.logs())
            nt = sum(1 for e in ev if e[0] == "w" and (e[2] == "CPU" or e[2].startswith("CPU")))
            npu = sum(1 for e in ev if e[0] == "w" and not (e[2] == "CPU" or e[2].startswith("CPU")))
            if nt >= want_take and npu >= want_put:
                return True
            time.sleep(1)
        return False

    def run(self):
        a = self.a
        res = {"model": self.model, "k": self.k, "graphs_off": self.graphs_off}
        self.start()
        if a.dry_run:
            self.stop()
            return res
        try:
            res["vram_idle_mib"] = vram_used_mib(a.vram_path)
            # 1. warm-up request: loads the model, and the LLAMA_FLEX_TEST takes happen after its first tokens
            t0 = time.time()
            warm = self.gen("Say hello in one short sentence.", 24)
            res["load_s"] = round(time.time() - t0, 1)
            res["warmup_text"] = warm.get("response", "")
            ok_moves = self.wait_moves(self.k, 0, 60) if self.k else True
            time.sleep(3)
            res["vram_after_take_mib"] = vram_used_mib(a.vram_path)
            res["takes_done"] = ok_moves
            # 2. measurements with the layers off
            gens, pps = [], []
            text = ""
            for i in range(a.reps):
                r = self.gen("Write a detailed explanation of how a bicycle works. Take care to mention the chain.", a.n_gen)
                gens.append(self.tps(r, "eval"))
                if i == 0:
                    text = r.get("response", "")
            for i in range(a.reps):
                r = self.gen(long_prompt("%s-%d-%d" % (self.name, i, int(time.time())), a.long_words), 1)
                pps.append(self.tps(r, "prompt_eval"))
                res["long_prompt_tokens"] = r.get("prompt_eval_count")
            res["gen_tps"] = round(statistics.median(gens), 2)
            res["pp_tps"] = round(statistics.median(pps), 2)
            res["gen_text"] = text
            chk = self.gen("Count from 1 to 30, separated by commas. Output only the numbers.", 120)
            res["count_text"] = chk.get("response", "").strip()
            res["sane"] = bool(re.match(r"^\s*1\s*,\s*2\s*,\s*3\s*,", res["count_text"])) and "30" in res["count_text"]
            # 3. put back: filler requests until the LLAMA_FLEX_TEST puts (at --put-at tokens) happened
            if self.k:
                for i in range(a.max_filler):
                    if self.wait_moves(self.k, self.k, 1):
                        break
                    self.gen(long_prompt("filler-%s-%d-%d" % (self.name, i, int(time.time())), a.long_words), 1)
                res["puts_done"] = self.wait_moves(self.k, self.k, 30)
                time.sleep(3)
                res["vram_after_put_mib"] = vram_used_mib(a.vram_path)
                chk2 = self.gen("Count from 1 to 30, separated by commas. Output only the numbers.", 120)
                res["count_text_after_put"] = chk2.get("response", "").strip()
                res["same_after_put"] = res["count_text_after_put"] == res["count_text"]
            ev = parse_log(self.logs())
            res["moves"] = summarize_moves(ev)
            # the freed VRAM, also from the log: device free memory right after the last take
            takes = res["moves"]["take"]
            res["log_free_after_take_mib"] = max((d["free_mib"] for d in takes.values() if d["free_mib"] is not None), default=None)
            return res
        finally:
            self.stop()


# ---------------------------------------------------------------- report

def fmt(v, nd=1):
    return "-" if v is None else ("%.*f" % (nd, v))


def report(results, a, ref):
    L = []
    L.append("# Flex band NAS measurement\n")
    L.append("Image: `%s`, context %d, band %d layers, dual-buffer %s, long prompt ~%s tokens\n" %
             (a.image, a.ctx, a.band, "on" if a.dual_buffer else "off", next((str(r.get("long_prompt_tokens")) for r in results if r.get("long_prompt_tokens")), "?")))
    for model in a.models:
        L.append("\n## %s\n" % model)
        L.append("| taken layers | HIP graphs | gen tok/s | pp tok/s | VRAM idle MiB | VRAM used after take MiB | freed MiB (sysfs) | "
                 "take ms (w+K/V) | put ms (w+K/V) | splits after take | sane | output vs k=0 | graphs on/off output |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        base_used = {}
        for r in results:
            if r["model"] == model and r["k"] == 0:
                base_used[r["graphs_off"]] = r.get("vram_after_take_mib")
        for r in results:
            if r["model"] != model:
                continue
            if "gen_tps" not in r:
                L.append("| %d | %s | failed | | | | | | | | | | |" % (r["k"], "off" if r["graphs_off"] else "on"))
                continue
            freed = None
            b = base_used.get(r["graphs_off"])
            if b is not None and r.get("vram_after_take_mib") is not None:
                freed = b - r["vram_after_take_mib"]
            mv = r.get("moves", {})
            take_ms = sum(d["w_ms"] + d["kv_ms"] for d in mv.get("take", {}).values())
            put_ms = sum(d["w_ms"] + d["kv_ms"] for d in mv.get("put", {}).values())
            sp = [s for kind, s in mv.get("splits_after_move", []) if kind == "take"]
            ref_t = ref.get((model, 0, r["graphs_off"]))
            same_ref = "-" if (r["k"] == 0 or ref_t is None) else ("same" if r["gen_text"] == ref_t else "DIFFERENT")
            other = next((x for x in results if x["model"] == model and x["k"] == r["k"] and x["graphs_off"] != r["graphs_off"] and "gen_text" in x), None)
            same_g = "-" if other is None else ("same" if other["gen_text"] == r["gen_text"] else "DIFFERENT")
            L.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["k"], "off" if r["graphs_off"] else "on", fmt(r["gen_tps"]), fmt(r["pp_tps"]),
                fmt(r.get("vram_idle_mib"), 0), fmt(r.get("vram_after_take_mib"), 0), fmt(freed, 0),
                fmt(take_ms) if r["k"] else "-", fmt(put_ms) if r["k"] else "-",
                (sp[-1] if sp else "-"), "yes" if r.get("sane") else "NO", same_ref, same_g))
        L.append("")
        L.append("Notes: freed MiB = VRAM used with k=0 minus with k taken layers (same HIP graph setting). Take/put ms are summed "
                 "over the layers (weights + K/V); splits = the last reserve line after a take (pp / tg graph).")
    L.append("\n## Raw per-case details\n")
    for r in results:
        L.append("- `%s k=%d graphs_off=%s`: %s" % (r["model"], r["k"], r["graphs_off"],
                 json.dumps({k: v for k, v in r.items() if k not in ("gen_text", "warmup_text", "count_text", "count_text_after_put")}, default=str)))
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default="ghcr.io/csaki89/ollama-dualbuf:latest", help="image (pin the tag of the build under test)")
    p.add_argument("--models", default="qwen3-vl:8b,qwen3-coder:30b", help="comma separated Ollama model names")
    p.add_argument("--layers", default="0,1,2", help="numbers of layers taken off (each <= --band)")
    p.add_argument("--band", type=int, default=2, help="--flex-layers (size of the flex band)")
    p.add_argument("--num-gpu", default="", help="per model num_gpu for the request, e.g. qwen3-vl:8b=30 (default: Ollama decides; leave empty for the MoE model)")
    p.add_argument("--models-dir", default="/mnt/nvme/ollama/models")
    p.add_argument("--render-node", default="/dev/dri/renderD129")
    p.add_argument("--vram-path", default="/sys/bus/pci/devices/0000:86:00.0/mem_info_vram_used")
    p.add_argument("--vram-idle-max", type=float, default=300.0, help="warn if the GPU holds more than this many MiB before a case")
    p.add_argument("--memory", default="100g")
    p.add_argument("--cpus", type=int, default=60)
    p.add_argument("--port", type=int, default=30070)
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--reps", type=int, default=2, help="repetitions of each speed measurement (median)")
    p.add_argument("--n-gen", type=int, default=128)
    p.add_argument("--long-words", type=int, default=1500, help="size of the long prompt in words (~1.3 tokens each)")
    p.add_argument("--put-at", type=int, default=20000, help="token count at which LLAMA_FLEX_TEST puts the layers back")
    p.add_argument("--max-filler", type=int, default=30, help="max filler requests while waiting for the put-back")
    p.add_argument("--no-dual-buffer", dest="dual_buffer", action="store_false")
    p.add_argument("--no-graphs-off", dest="graphs_off", action="store_false", help="skip the GGML_CUDA_DISABLE_GRAPHS=1 runs")
    p.add_argument("--out", default="flex-bench-" + time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--dry-run", action="store_true", help="print the docker commands only")
    a = p.parse_args()
    a.models = [m.strip() for m in a.models.split(",") if m.strip()]
    a.layers = [int(x) for x in a.layers.split(",") if x.strip()]
    a.num_gpu = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.num_gpu.split(",") if "=" in kv}
    if any(k > a.band for k in a.layers):
        sys.exit("--layers values must not exceed --band")
    os.makedirs(a.out, exist_ok=True)

    results, ref = [], {}
    for model in a.models:
        for graphs_off in ([False, True] if a.graphs_off else [False]):
            for k in sorted(a.layers, key=lambda x: (x != 0, x)):  # k=0 first: it is the reference
                if not a.dry_run:
                    used = vram_used_mib(a.vram_path)
                    if used is not None and used > a.vram_idle_max:
                        print("WARNING: the GPU already holds %.0f MiB (production model loaded?): VRAM figures are distorted" % used, file=sys.stderr)
                print("== %s k=%d graphs_off=%s" % (model, k, graphs_off), flush=True)
                c = Case(a, model, k, graphs_off)
                try:
                    r = c.run()
                except Exception as e:  # keep going, report the failure
                    print("   failed: %s" % e, file=sys.stderr)
                    r = {"model": model, "k": k, "graphs_off": graphs_off, "error": str(e)}
                    c.stop()
                if k == 0 and "gen_text" in r:
                    ref[(model, 0, graphs_off)] = r["gen_text"]
                results.append(r)
                if "gen_tps" in r:
                    print("   gen %.1f tok/s, pp %.1f tok/s, sane=%s" % (r["gen_tps"], r["pp_tps"], r.get("sane")), flush=True)
    if a.dry_run:
        return
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    md = report(results, a, ref)
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write(md)
    print(md)
    print("Written to %s (results.md, results.json, logs/)" % a.out)


if __name__ == "__main__":
    main()
