# Input schema summary

| Field family | Purpose |
|---|---|
| `id` | Trip join and split-audit key; excluded from predictors |
| `label`, `subscription_type` | Customer=1 / Subscriber=0; target text excluded from predictors |
| `duration_sec` | Original trip duration in seconds; Spark defaults to a log transform |
| `start_hour`, `day_of_week`, `month` | Verified original local-clock calendar fields |
| `is_weekend`, `is_holiday`, `is_round_trip` | Calendar and trip flags |
| Start/end station IDs | Join/audit keys; not the raw classifier identifiers |
| `start_city`, `trip_date` | Station/weather join fields; city is a paper-reconstruction category |
| `start_cluster`, `end_cluster` | k=5 static-station coordinate clusters |
| Daily temperature, dew point, humidity, pressure, visibility, wind, precipitation, cloud | Weather predictors; missing numeric values imputed inside classifier fitting folds |
| `precipitation_trace`, weather event indicators | Explicit handling of trace precipitation and events |
| `is_business_day`, `is_inter_cluster` | Derived flags |
| `start_ts` | Legacy timestamp encoding; excluded from predictors |
| `local_zip` | Derived rider locality for the separately labeled paper reconstruction; excluded from proposed P1-P3 |

The paper-visible reconstruction adds `before_oct_2013` and a combined fog/rain indicator deterministically. The exact original locality and abbreviated weather aggregations are unknown; the assumptions are documented in the reconstruction guide. Model manifests list the exact predictor columns used for every saved run.

Schema documentation contains field definitions only. No individual trip IDs, timestamps, locations, postal codes, or prediction rows are published here.
