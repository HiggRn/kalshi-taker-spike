import json
import sys
from collections import defaultdict
from itertools import chain, groupby
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RAW_DIR, OUT_DIR = Path("raw_data"), Path("data")
PRICE_SCALE = 1000
SIZE_SCALE = 100
MS_NS = 1_000_000
DUST = 1
MIN_VERIFIED = 1
SHOW = 5

cat = pa.dictionary(pa.int8(), pa.string())
SCHEMA = pa.schema(
    [
        ("event_seq", pa.uint64()),
        ("timestamp_ns", pa.int64()),  # exchange ts_ms * 1e6 (snapshot: see build_rows)
        ("recv_ns", pa.int64()),  # local receive time, kept for latency analysis
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
        ("segment", cat),  # one per verified snapshot-delimited segment
    ]
)


def px(s):
    return round(float(s) * PRICE_SCALE)


def sz(s):
    return round(float(s) * SIZE_SCALE)


def levels(pairs):
    """[["0.9030","50000.00"], ...] -> {903: 5000000}"""
    return {px(p): sz(s) for p, s in pairs if sz(s) > 0}


def apply_delta(side, price, delta):
    """Keep the raw running sum (may be transiently <= 0 when events are
    reordered); only an exact zero is removed. add_row/strip hide non-positive."""
    new_size = side.get(price, 0) + delta
    if new_size == 0:
        side.pop(price, None)
    else:
        side[price] = new_size


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


def add_row(
    rows, last_ts, ticker, ts, recv_ns, event_type, book, trade=(None, None, None)
):
    """book = {"yes": {price: size}, "no": {price: size}}"""
    bids = sorted(((p, s) for p, s in book["yes"].items() if s > 0), reverse=True)
    asks = sorted((PRICE_SCALE - p, s) for p, s in book["no"].items() if s > 0)
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    spread = best_ask - best_bid if best_bid and best_ask else None
    prev = last_ts.get(ticker)
    rows.append(
        {
            "event_seq": len(rows),
            "timestamp_ns": ts,
            "recv_ns": recv_ns,
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


# ---------------------------------------------------------------------------
# Event ordering:
#   1. exchange ts_ms
#   2. within the same ts_ms: trades first (recv_ns order), then orderbook deltas
#   3. deltas go from the top of the book to deeper levels: depth = number of
#      existing levels on that side strictly better than the delta's price,
#      measured on the book as it stands before the ts_ms group is applied
#      (depth 0 = best level, or a new level that would become the best)
#   4. at equal depth bid (yes) comes before ask (no); then better price first;
#      recv_ns as the final tiebreak
# ---------------------------------------------------------------------------
SIDE_RANK = {"yes": 0, "no": 1}  # yes side = bids, no side = asks


def base_key(rec):
    d = rec["data"]
    return (d["msg"]["ts_ms"], d["type"] != "trade", rec["recv_ns"])


def order_deltas(deltas, book):
    """Sort one ts_ms group's deltas from top of book down (bid before ask)."""

    def key(rec):
        m = rec["data"]["msg"]
        side, price = m["side"], px(m["price_dollars"])
        # on both sides a higher raw price is closer to the top of the book
        depth = sum(1 for p, s in book[side].items() if s > 0 and p > price)
        return (depth, SIDE_RANK[side], -price, rec["recv_ns"])

    return sorted(deltas, key=key)


def build_rows(seg):
    tk = seg["ticker"]
    snap = seg["snap"]
    sm = snap["data"]["msg"]
    events = sorted(seg["events"], key=base_key)
    book = {
        "yes": levels(sm.get("yes_dollars_fp", [])),
        "no": levels(sm.get("no_dollars_fp", [])),
    }
    rows, last_ts = [], {}

    # The snapshot message carries no exchange timestamp; use its recv time but
    # never later than the first (sorted) event, so timestamp_ns stays monotone.
    snap_ts = snap["recv_ns"]
    if events:
        snap_ts = min(snap_ts, events[0]["data"]["msg"]["ts_ms"] * MS_NS)
    add_row(rows, last_ts, tk, snap_ts, snap["recv_ns"], "snapshot", book)

    for ts_ms, grp in groupby(events, key=lambda r: r["data"]["msg"]["ts_ms"]):
        grp = list(grp)
        ts = ts_ms * MS_NS
        trades = [r for r in grp if r["data"]["type"] == "trade"]
        deltas = [r for r in grp if r["data"]["type"] != "trade"]
        for rec in trades:
            m = rec["data"]["msg"]
            trade = (
                px(m["yes_price_dollars"]),
                sz(m["count_fp"]),
                m["taker_side"] + "_taker",
            )
            add_row(rows, last_ts, tk, ts, rec["recv_ns"], "trade", book, trade)
        for rec in order_deltas(deltas, book):
            m = rec["data"]["msg"]
            apply_delta(book[m["side"]], px(m["price_dollars"]), sz(m["delta_fp"]))
            add_row(rows, last_ts, tk, ts, rec["recv_ns"], "delta_book", book)
    return rows


def write_segment(seg):
    stat = seg["stat"]
    n = len(seg["events"]) + 1
    if stat["bad"] or stat["ok"] < MIN_VERIFIED:
        print(
            f"  SKIP  {stat['label']}: {n} rows, checkpoints ok={stat['ok']} bad={stat['bad']} (need bad=0 and ok>={MIN_VERIFIED})"
        )
        return False
    if not seg["events"]:
        return None  # snapshot only: nothing to model
    rows = build_rows(seg)
    for r in rows:
        r["segment"] = stat["label"]
    print(f"  OK    {stat['label']}: {len(rows)} rows, checkpoints ok={stat['ok']}")
    return pa.Table.from_pylist(rows, schema=SCHEMA)


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
    # Pass 1 (recv order): checkpoints are stamped with asof_recv_ns, so they must
    # be verified against the book replayed in receive order. Events are only
    # collected per snapshot-delimited segment here; they are re-sorted by ts_ms
    # when rows are built (deltas are additive, so the end state is the same).
    books, cur, stats, segments = {}, {}, {}, []
    c = 0

    for rec in chain.from_iterable(read_records(p) for p in paths):
        d = rec.get("data", {})
        t, m, ts = d.get("type"), d.get("msg", {}), rec["recv_ns"]
        tk = m.get("market_ticker")

        while c < len(ckpts) and ckpts[c][1] < ts:
            check(ckpts[c], books, stats)
            c += 1

        if t == "orderbook_snapshot":
            label = f"{tk}_{ts}" if tk in books else f"{tk}_{recorder_ns}"
            seg = {"ticker": tk, "stat": new_stat(label), "snap": rec, "events": []}
            segments.append(seg)
            cur[tk] = seg
            stats[tk] = seg["stat"]
            books[tk] = {
                "yes": levels(m.get("yes_dollars_fp", [])),
                "no": levels(m.get("no_dollars_fp", [])),
            }

        elif tk not in books:
            continue

        elif t == "orderbook_delta":
            apply_delta(books[tk][m["side"]], px(m["price_dollars"]), sz(m["delta_fp"]))
            cur[tk]["events"].append(rec)

        elif t == "trade":
            cur[tk]["events"].append(rec)

    for ckpt in ckpts[c:]:
        check(ckpt, books, stats)

    # Pass 2: sort each verified segment by ts_ms into a table.
    tables = [write_segment(seg) for seg in segments]
    return [t for t in tables if isinstance(t, pa.Table)], tables.count(False)


def combine(tables):
    """Concatenate one ticker's segments in time order. Where segments overlap
    (two recorders running at once), cut the earlier one at the later one's
    snapshot: the later book is exact from its snapshot, the earlier copy of
    the overlap is a duplicate."""
    tables.sort(key=lambda t: t["timestamp_ns"][0].as_py())
    parts = [
        a.filter(pc.less(a["timestamp_ns"], b["timestamp_ns"][0]))
        for a, b in zip(tables, tables[1:])
    ] + tables[-1:]
    out = pa.concat_tables(parts)
    return out.set_column(0, "event_seq", pa.array(range(len(out)), pa.uint64()))


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    skipped = 0
    groups = defaultdict(list)  # (recorder, ticker) -> [segments...]
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        if path.stem.endswith("__book-checkpoints"):
            continue
        _, recorder, _, ticker = path.stem.split("__")
        groups[(recorder, ticker)].append(path)
    by_ticker = defaultdict(list)
    for paths in groups.values():
        print(", ".join(p.name for p in paths))
        tables, n_skip = process_message_file(paths)
        skipped += n_skip
        for t in tables:
            by_ticker[t["ticker"][0].as_py()].append(t)
    for tk, tables in sorted(by_ticker.items()):
        out = combine(tables)
        pq.write_table(out, OUT_DIR / f"{tk}.parquet", compression="zstd")
        print(f"WRITE {tk}: {len(out)} rows, {len(tables)} segments")
    if skipped:
        print(f"\n{skipped} parts are skipped due to checkpoint mismatch.")
        sys.exit(1)
