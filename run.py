#!/usr/bin/env python3
"""Chip404 free-lane PoW driver for Kaggle GPU notebooks.

Compiles solver.cu, runs one shard per GPU, watches the chain, and writes
result.json to /kaggle/working (or ./out locally) as soon as a usable nonce
is found. The nonce is always verified locally (pure-python keccak) against
the freshest targetNow() before being reported.
"""
import json, os, struct, subprocess, sys, time, urllib.request

WALLET = "7b6c82c0dfabc16882674da1057f01721e686791"
CONTRACT = "0xAB6b18C626daaDd1ff7Cc45F402141153a5a9759"
RPCS = ["https://bsc-dataseed1.defibit.io", "https://bsc-dataseed2.defibit.io",
        "https://bsc-dataseed3.defibit.io", "https://bsc-dataseed.binance.org",
        "https://bsc-dataseed1.bnbchain.org", "https://bsc.publicnode.com",
        "https://bsc.meowrpc.com", "https://binance.llamarpc.com"]
# Snapshot fallback (2026-11-02 06:37 WIB): weight 43, totalMinted 1758.
SNAPSHOT = {"target": "00000000000fffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "weight": 44, "minted": 1859}
OVERALL_CAP_SEC = int(os.environ.get("CHIP404_CAP_SEC", "19800"))  # 5.5 h
EPOCH = 150

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.getcwd(), "out")
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- pure-python keccak-256 (Keccak pad 0x01, not SHA3) ----------------
RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
      0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
      0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
      0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
      0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
      0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
PILN = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]
MASK = (1 << 64) - 1

def _rol(x, n):
    return ((x << n) | (x >> (64 - n))) & MASK

def _keccakf(st):
    for rnd in range(24):
        bc = [st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20] for i in range(5)]
        for i in range(5):
            t = bc[(i + 4) % 5] ^ _rol(bc[(i + 1) % 5], 1)
            for j in range(0, 25, 5):
                st[j + i] ^= t
        t = st[1]
        for i in range(24):
            j = PILN[i]
            t, st[j] = st[j], _rol(t, ROTC[i])
        for j in range(0, 25, 5):
            row = st[j:j + 5]
            for i in range(5):
                st[j + i] = row[i] ^ ((~row[(i + 1) % 5]) & row[(i + 2) % 5])
        st[0] ^= RC[rnd]

def keccak256(data: bytes) -> bytes:
    rate = 136
    msg = bytearray(data)
    pad_len = rate - (len(msg) % rate)
    if pad_len == 1:
        msg.append(0x81)
    else:
        msg.append(0x01)
        msg.extend(b"\x00" * (pad_len - 2))
        msg.append(0x80)
    st = [0] * 25
    for off in range(0, len(msg), rate):
        blk = msg[off:off + rate]
        for i in range(rate // 8):
            st[i] ^= struct.unpack("<Q", blk[i * 8:(i + 1) * 8])[0]
        _keccakf(st)
    out = b"".join(struct.pack("<Q", st[i]) for i in range(4))
    return out[:32]

SALT = keccak256(b"chip404.trial.v1")
SENDER = bytes.fromhex(WALLET)

def chip_hash(nonce: int) -> bytes:
    return keccak256(SALT + SENDER + nonce.to_bytes(32, "big"))

def lz_bits(h: bytes) -> int:
    v = int.from_bytes(h, "big")
    return 256 - v.bit_length() if v else 256

# ---------------- chain reads ----------------
def _sel(sig: str) -> str:
    return "0x" + keccak256(sig.encode())[:4].hex()

def eth_call(data_hex: str):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                       "params": [{"to": CONTRACT, "data": data_hex}, "latest"]}).encode()
    for rpc in RPCS:
        try:
            req = urllib.request.Request(rpc, data=body,
                                         headers={"Content-Type": "application/json",
                                                  "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                res = json.load(r)
            if res.get("result"):
                return res["result"]
        except Exception:
            continue
    return None

def chain_state():
    t = eth_call(_sel("targetNow()"))
    w = eth_call(_sel("weightNow()"))
    m = eth_call(_sel("totalMinted()"))
    if t and w and m:
        return {"target": t[2:].rjust(64, "0"), "weight": int(w, 16), "minted": int(m, 16)}
    print("[driver] RPC unreachable, using snapshot", flush=True)
    return dict(SNAPSHOT)

# ---------------- solver management ----------------
def find_nvcc():
    for cand in ["nvcc", "/usr/local/cuda/bin/nvcc"]:
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except Exception:
            continue
    return None

def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return max(1, out.stdout.count("GPU "))
    except Exception:
        return 1

def read_hit(path):
    try:
        with open(path) as f:
            kv = dict(line.split("=", 1) for line in f.read().strip().splitlines())
        return int(kv["counter"]), int(kv["shard"])
    except Exception:
        return None

def finalize(nonce, status, state):
    h = chip_hash(nonce)
    rec = {"status": status, "nonce": str(nonce), "nonce_hex": format(nonce, "064x"),
           "hash": h.hex(), "lz": lz_bits(h), "weight": state["weight"],
           "totalMinted": state["minted"], "target": state["target"],
           "found_at_unix": int(time.time())}
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print("\n================ CHIP404 RESULT ================", flush=True)
    print(json.dumps(rec, indent=2), flush=True)
    print("================================================\n", flush=True)
    return rec

def main():
    nvcc = find_nvcc()
    if not nvcc:
        print("[driver] FATAL: nvcc not found (is a GPU accelerator selected?)", flush=True)
        return 10
    binary = os.path.join(HERE, "solver")
    cmd = [nvcc, "-O3", "-std=c++14",
           "-gencode", "arch=compute_60,code=sm_60",
           "-gencode", "arch=compute_70,code=sm_70",
           "-gencode", "arch=compute_75,code=sm_75",
           os.path.join(HERE, "solver.cu"), "-o", binary]
    print("[driver] compiling:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
        return 11
    ngpu = gpu_count()
    print(f"[driver] GPUs: {ngpu}", flush=True)

    t_start = time.time()
    round_no = 0
    fallback = None  # (nonce, found_unix)
    while time.time() - t_start < OVERALL_CAP_SEC:
        round_no += 1
        state = chain_state()
        target_int = int(state["target"], 16)
        thr_plain = format(target_int, "064x")
        thr_margin = format(target_int >> 2, "064x")
        print(f"[driver] round {round_no}: weight={state['weight']} minted={state['minted']} "
              f"target={thr_plain[:20]}...", flush=True)
        remain = int(OVERALL_CAP_SEC - (time.time() - t_start))
        procs, prefixes = [], []
        for g in range(ngpu):
            shard = (round_no - 1) * ngpu + g
            prefix = os.path.join(OUT, f"shard{shard}")
            prefixes.append(prefix)
            for suffix in ("_plain.txt", "_margin.txt"):
                p = prefix + suffix
                if os.path.exists(p):
                    os.remove(p)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            args = [binary, SALT.hex(), WALLET, thr_plain, thr_margin,
                    str(shard), str(remain), prefix, "1200"]
            log = open(prefix + ".log", "w")
            procs.append(subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env))
        print(f"[driver] launched shards {[ (round_no-1)*ngpu + g for g in range(ngpu)]}", flush=True)

        round_done = False
        while not round_done:
            time.sleep(20)
            # margin hit anywhere -> finalize now
            for prefix in prefixes:
                hit = read_hit(prefix + "_margin.txt")
                if hit:
                    counter, shard = hit
                    nonce = (shard << 64) | counter
                    st = chain_state()
                    if int.from_bytes(chip_hash(nonce), "big") <= int(st["target"], 16):
                        for p in procs:
                            p.kill()
                        finalize(nonce, "margin", st)
                        return 0
                    print("[driver] margin hit failed fresh-target verify (epoch moved twice?)", flush=True)
            # plain hits -> maybe sprint
            st = chain_state()
            for prefix in prefixes:
                hit = read_hit(prefix + "_plain.txt")
                if not hit:
                    continue
                counter, shard = hit
                nonce = (shard << 64) | counter
                h_ok = int.from_bytes(chip_hash(nonce), "big") <= int(st["target"], 16)
                if not h_ok:
                    continue
                if fallback is None or fallback[0] != nonce:
                    fallback = (nonce, time.time())
                    print(f"[driver] plain candidate nonce={nonce} passes fresh target "
                          f"(minted={st['minted']}, weight={st['weight']})", flush=True)
                dist = EPOCH - (st["minted"] % EPOCH)
                age = time.time() - fallback[1]
                if dist <= 80 or age >= 600:
                    for p in procs:
                        p.kill()
                    finalize(nonce, "plain-sprint", st)
                    return 0
            if all(p.poll() is not None for p in procs):
                round_done = True
            if time.time() - t_start >= OVERALL_CAP_SEC:
                break
        # round ended without a take: if a fallback still passes, use it
        if fallback:
            st = chain_state()
            if int.from_bytes(chip_hash(fallback[0]), "big") <= int(st["target"], 16):
                finalize(fallback[0], "plain-at-cap", st)
                return 0
            print("[driver] fallback nonce went stale, continuing", flush=True)
            fallback = None
    print("[driver] overall cap reached without a passing nonce", flush=True)
    return 5

if __name__ == "__main__":
    sys.exit(main())
