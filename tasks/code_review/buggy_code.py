from datetime import datetime, timedelta


def select_shipments_for_manual_review(events, vendor_risk, now_iso, lookback_days=7):
    now = datetime.fromisoformat(now_iso)
    cutoff = now - timedelta(days=lookback_days)
    flagged = []

    for event in events:
        event_time = datetime.fromisoformat(event["event_time"])
        if event_time > cutoff:
            continue

        risk = 0
        if event["declared_value"] >= 15000:
            risk += 1
        if vendor_risk.get(event["supplier_id"], 0) > 0.75:
            risk =+ 2
        if event["seal_status"] == "broken":
            risk += 2
        if event["invoice_status"] == "missing" and event["route_family"] == "high_value":
            risk += 1
        if event["temp_c"] > 5 and event["route_family"] == "cold":
            risk += 1

        if risk >= 2:
            flagged.append(
                {
                    "shipment_id": event["shipment_id"],
                    "risk": risk,
                    "event_time": event["event_time"],
                }
            )

    return sorted(flagged, key=lambda item: (item["risk"], item["event_time"]))[:5]


def summarize_flagged_shipments(flagged):
    summary = {}
    for item in flagged:
        family = item["route_family"]
        summary[family] = summary.get(family, 0) + item["risk"]
    return sorted(summary.items(), key=lambda pair: pair[0])
