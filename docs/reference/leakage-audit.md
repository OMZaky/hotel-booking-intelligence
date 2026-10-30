# Preliminary leakage audit

Status: hypotheses for review; no approved feature list or completed data audit yet.

## Source timing

The [original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/) describes reconstructing
many booking/change-log values relative to the day before arrival. These snapshots are not a
full booking-creation event history. ADR is transaction-derived; DepositType is based on payments
before arrival/cancellation. Neither automatically represents an original quote or deposit policy.

Define prediction points separately for ADR and cancellation before training.
Day-before-arrival cancellation prediction is also problematic for records already canceled by then:
do not describe an already known outcome as a future prediction.
Booking-creation prediction needs conservative availability assumptions and disclosed limitations.
No split strategy repairs predictors observed after the claimed prediction point.

## Initial review

| Features | Risk | Initial action |
| --- | --- | --- |
| ADR | Regression target; transaction-derived | Never regression input; classification needs timing evidence; omit from initial clustering |
| IsCanceled | Final cancellation outcome | Classification target only; exclude from regression/clustering |
| ReservationStatus, ReservationStatusDate | Final status/outcome timing | Exclude all predictor matrices; use separately only for justified auditing/label availability |
| AssignedRoomType | May be assigned after booking creation | Quarantine pending timing review |
| BookingChanges | Accumulated lifecycle amendments | Quarantine for booking-creation prediction |
| DaysInWaitingList | Confirmation/waiting outcome | Review operational availability |
| DepositType | Derived from payments through arrival/cancellation | Quarantine pending timing review |
| RequiredCarParkingSpaces, TotalOfSpecialRequests | Requests may evolve | Review snapshot availability |
| PreviousCancellations, PreviousBookingsNotCanceled, IsRepeatedGuest | Historical-looking aggregates need as-of definitions | Review source construction and disclose limitations |
| Dates, nights, party size, reserved room/meal | Plausible initial inputs, but may be amended | Review snapshot assumptions explicitly |
| Agent, Company, Country, MarketSegment, DistributionChannel, CustomerType | IDs/categories and evolving values | Review timing, missingness, encoding and cardinality |

## Complete audit before experiments

- Approve a per-task allowlist recording name, meaning, availability, rationale and decision.
- Preserve exact source labels; document a one-to-one mapping if renamed.
- Freeze split/cohort IDs and duplication/group policies.
- Fit learned preprocessing within training folds, including representation and feature selection.
- Trace engineered features to inputs; exclude transformations of banned columns.
- Tune on CV/validation only; keep final test data sealed.
- Keep outcomes out of clustering features, optimization and unsupervised scores.
- Save feature policy and input schema with each final pipeline.
- Freeze selection before final test evaluation; disclose subsequent corrections.

The config validator catches obvious target/final-status names only.
It cannot prove timing, prevent indirect leakage, or approve an omitted feature list.
