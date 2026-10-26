import json
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR, OUT_DIR = Path("raw_data"), Path("data")
PRICE_SCALE = 1000
SIZE_SCALE = 100
DUST = 1
MIN_VERIFIED = 1
SHOW = 5

cat = pa.dictionary(pa.int8(), pa.string())
SCHEMA = pa.schema(
    [
        ("event_seq", pa.uint64()),
        ("timestamp_ns", pa.int64()),
        ("delta_t_ms", pa.float32()),
        ("ticker", cat),
        ("event_type", cat),
        ("bid_prices", pa.list_(pa.int32())),
        ("bid_sizes", pa.list_(pa.int64())),
        ("ask_prices", pa.list_(pa.int32())),
        ("ask_sizes", pa.list_(pa.int64())),
        ("best_bid", pa.int32()),
        ("best_ask", pa.int32()),
        ("spread", pa.int32()),
        ("trade_price", pa.int32()),
        ("trade_size", pa.int64()),
        ("taker_side", cat),
    ]
)


def px(s):
    return round(float(s) * PRICE_SCALE)


def sz(s):
    return round(float(s) * SIZE_SCALE)


def levels(pairs):
    """[["0.9030","50000.00"], ...] -> {903: 5000000}"""
    return {px(p): sz(s) for p, s in pairs if sz(s) > 0}


def read_records(path):
    buf = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if buf and not line.startswith(","):
                yield from parse(buf)
                buf = ""
            buf += line
    if buf:
        yield from parse(buf)


def parse(text):
    try:
        yield json.loads(text)
    except json.JSONDecodeError:
        pass


def load_checkpoints(path):
    """[(ticker, asof_recv_ns, book)] -> {"yes": {price: size}, "no": {...}}"""
    if not path.exists():
        print(f"  WARNING: Can't find {path.name}. This file won't be processed.")
        return []
    ckpts = []
    for rec in read_records(path):
        ob = rec["orderbook"]
        book = {"yes": levels(ob["yes_dollars_fp"]), "no": levels(ob["no_dollars_fp"])}
        ckpts.append((rec["market_ticker"], rec["asof_recv_ns"], book))
    return sorted(ckpts, key=lambda x: x[1])


def strip(book):
    return {
        side: {p: s for p, s in lv.items() if s > DUST} for side, lv in book.items()
    }


def check(ckpt, books, stats):
    ticker, asof, ck_book = ckpt
    if ticker not in books:
        return
    stat = stats[ticker]
    a, b = strip(ck_book), strip(books[ticker])
    if a == b:
        stat["ok"] += 1
        return
    stat["bad"] += 1
    if stat["bad"] <= SHOW:
        diffs = [
            (side, p, a[side].get(p), b[side].get(p))
            for side in ("yes", "no")
            for p in sorted(a[side].keys() | b[side].keys())
            if a[side].get(p) != b[side].get(p)
        ]
        print(
            f"  MISMATCH {stat['label']} {ticker} asof={asof}  (side, price, ckpt_size, replay_size): {diffs[:10]}"
        )


def add_row(rows, last_ts, ticker, ts, event_type, book, trade=(None, None, None)):
    """book = {"yes": {price: size}, "no": {price: size}}"""
    bids = sorted(book["yes"].items(), reverse=True)
    asks = sorted((PRICE_SCALE - p, s) for p, s in book["no"].items())
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    spread = best_ask - best_bid if best_bid and best_ask else None
    prev = last_ts.get(ticker)
    rows.append(
        {
            "event_seq": len(rows),
            "timestamp_ns": ts,
            "delta_t_ms": None if prev is None else (ts - prev) / 1e6,
            "ticker": ticker,
            "event_type": event_type,
            "bid_prices": [p for p, _ in bids],
            "bid_sizes": [s for _, s in bids],
            "ask_prices": [p for p, _ in asks],
            "ask_sizes": [s for _, s in asks],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "trade_price": trade[0],
            "trade_size": trade[1],
            "taker_side": trade[2],
        }
    )
    last_ts[ticker] = ts


def new_stat(label):
    return {"label": label, "ok": 0, "bad": 0}


def write_part(rows, stat):
    if not rows:
        return True
    if stat["bad"] or stat["ok"] < MIN_VERIFIED:
        print(
            f"  SKIP  {stat['label']}: {len(rows)} rows, checkpoints ok={stat['ok']} bad={stat['bad']} (need bad=0 and ok>={MIN_VERIFIED})"
        )
        return False
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA),
        OUT_DIR / f"{stat['label']}.parquet",
        compression="zstd",
    )
    print(f"  WRITE {stat['label']}: {len(rows)} rows, checkpoints ok={stat['ok']}")
    return True


def process_message_file(paths):
    recorder_ns = paths[0].stem.split("__")[1].removeprefix("recorder-")
    ckpts = sorted(
        (
            c
            for p in paths
            for c in load_checkpoints(p.with_name(f"{p.stem}__book-checkpoints.jsonl"))
        ),
        key=lambda x: x[1],
    )
    books, last_ts, rows, stats = {}, {}, defaultdict(list), {}
    c = skipped = 0

    for rec in chain.from_iterable(read_records(p) for p in paths):
        d = rec.get("data", {})
        t, m, ts = d.get("type"), d.get("msg", {}), rec["recv_ns"]
        tk = m.get("market_ticker")

        while c < len(ckpts) and ckpts[c][1] < ts:
            check(ckpts[c], books, stats)
            c += 1

        if t == "orderbook_snapshot":
            if tk in books:
                skipped += not write_part(rows[tk], stats[tk])
                rows[tk] = []
                stats[tk] = new_stat(f"{tk}_{ts}")
                last_ts.pop(tk, None)
            else:
                stats[tk] = new_stat(f"{tk}_{recorder_ns}")
            books[tk] = {
                "yes": levels(m.get("yes_dollars_fp", [])),
                "no": levels(m.get("no_dollars_fp", [])),
            }
            add_row(rows[tk], last_ts, tk, ts, "snapshot", books[tk])

        elif tk not in books:
            continue

        elif t == "orderbook_delta":
            side = books[tk][m["side"]]
            price = px(m["price_dollars"])
            new_size = side.get(price, 0) + sz(m["delta_fp"])
            if new_size > 0:
                side[price] = new_size
            else:
                side.pop(price, None)
            add_row(rows[tk], last_ts, tk, ts, "delta_book", books[tk])

        elif t == "trade":
            trade = (
                px(m["yes_price_dollars"]),
                sz(m["count_fp"]),
                m["taker_side"] + "_taker",
            )
            add_row(rows[tk], last_ts, tk, ts, "trade", books[tk], trade)

    for ckpt in ckpts[c:]:
        check(ckpt, books, stats)
    for tk in rows:
        skipped += not write_part(rows[tk], stats[tk])
    return skipped


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    skipped = 0
    groups = defaultdict(list)  # (recorder, ticker) -> [segments...]
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        if path.stem.endswith("__book-checkpoints"):
            continue
        _, recorder, _, ticker = path.stem.split("__")
        groups[(recorder, ticker)].append(path)
    for paths in groups.values():
        print(", ".join(p.name for p in paths))
        skipped += process_message_file(paths)
    if skipped:
        print(f"\n{skipped} parts are skipped due to checkpoint mismatch.")
        sys.exit(1)
