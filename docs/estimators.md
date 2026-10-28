# Estimators

The `Engine` asks an `Estimator` for each waiting job's estimate before every `decide`, so `decide` stays pure.

- `HistoryEstimator(store)`: learns peaks per fingerprint key (Rule H). It answers from the most specific key that has data. The answer is `max(complete records, newer lower bounds)`.
  - **Trusted** only when a complete record gives the value. A lower bound alone (from an OOM) says the job needs *more* than that, so it is untrusted.
  - Running jobs write a record when they become STABLE and when their peak grows. Finished jobs write their final peak.
- `NullEstimator()`: always unknown. Every job is untrusted and goes through probation.
- `StaticEstimator(mapping)`: fixed estimates, for tests.

## History stores

`HistoryStore` is a protocol: `get(key, device_kind)` and `put(key, device_kind, peak, lower_bound=..., record_id=...)`. A put with an existing `record_id` max-merges, so writing a record twice is harmless. `InMemoryHistoryStore` keeps everything in memory and has `to_json` / `from_json`. In production, back the protocol with your database: the store must be durable, because the Engine does not dump it.

Device kinds come from `Device.labels["kind"]` by default, so a template's history learned on one A100 applies to every A100.
