"""Predictive Intelligence Layer — Anomaly detection, decay prediction, forecasting."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_supabase():
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if url and key:
            return create_client(url, key)
    except ImportError:
        pass
    return None


class PredictiveLayer:
    """Detects anomalies, predicts decay, and forecasts trends."""

    def __init__(self):
        from .codex import PredictiveCodex
        self.codex = PredictiveCodex()
        self.supabase = _get_supabase()

    def detect_algorithm_update(self) -> Optional[Dict]:
        """Detect probable algorithm update: 3+ sites with >15% traffic change on same day."""
        if not self.supabase:
            return None

        try:
            # Get daily clicks for all sites (last 14 days)
            resp = self.supabase.table("gsc_performance") \
                .select("site_slug,date,clicks") \
                .order("date", desc=True) \
                .limit(500) \
                .execute()

            rows = resp.data or []
            if len(rows) < 50:
                return None

            # Group by date
            daily = {}
            for r in rows:
                date = r.get("date", "")
                site = r.get("site_slug", "")
                if date not in daily:
                    daily[date] = {}
                daily[date][site] = r.get("clicks", 0)

            # Compare consecutive days
            dates = sorted(daily.keys(), reverse=True)
            for i in range(len(dates) - 1):
                today = dates[i]
                yesterday = dates[i + 1]
                sites_affected = []

                for site in daily.get(today, {}):
                    today_clicks = daily[today].get(site, 0)
                    yest_clicks = daily[yesterday].get(site, 0)
                    if yest_clicks > 10:  # Minimum threshold
                        change = (today_clicks - yest_clicks) / yest_clicks * 100
                        if abs(change) > 15:
                            sites_affected.append({
                                "site": site,
                                "change_pct": round(change, 1),
                                "today": today_clicks,
                                "yesterday": yest_clicks,
                            })

                if len(sites_affected) >= 3:
                    anomaly = {
                        "type": "algorithm_update",
                        "date": today,
                        "sites_affected": len(sites_affected),
                        "details": sites_affected,
                    }
                    self.codex.log_anomaly(
                        "algorithm_update", "critical",
                        [s["site"] for s in sites_affected],
                        f"Probable algo update on {today}: {len(sites_affected)} sites "
                        f"with >15% traffic change",
                        anomaly
                    )

                    try:
                        from core.event_bus import publish
                        publish("predictive.algo_update_detected", anomaly, "predictive_layer")
                    except Exception:
                        pass

                    return anomaly

        except Exception as e:
            log.error(f"Algorithm detection error: {e}")

        return None

    def predict_decay(self, site_slug: str) -> List[Dict]:
        """Predict article decay using trailing click data."""
        if not self.supabase:
            return []

        predictions = []
        try:
            resp = self.supabase.table("top_pages") \
                .select("page_url,clicks,impressions") \
                .eq("site_slug", site_slug) \
                .order("clicks", desc=True) \
                .limit(100) \
                .execute()

            for row in (resp.data or []):
                clicks = row.get("clicks", 0)
                impressions = row.get("impressions", 0)
                url = row.get("page_url", "")

                if clicks < 5:
                    continue

                # Simple decay estimation: CTR-based
                ctr = clicks / max(impressions, 1)
                if ctr < 0.01 and impressions > 100:
                    # Low CTR with impressions = title/content issue, not decay
                    decay_rate = 0.05
                    action = "optimize_title"
                elif clicks < 10:
                    decay_rate = 0.3
                    action = "refresh_content"
                else:
                    decay_rate = 0.1
                    action = "monitor"

                predicted_30d = max(0, int(clicks * (1 - decay_rate)))
                days_to_zero = int(clicks / max(clicks * decay_rate / 30, 0.01))

                title = url.split("/")[-2].replace("-", " ").title() if "/" in url else url

                self.codex.log_decay(
                    site_slug, url, title, clicks,
                    predicted_30d, decay_rate, days_to_zero, action
                )

                if decay_rate > 0.15:
                    predictions.append({
                        "url": url,
                        "title": title,
                        "current_clicks": clicks,
                        "predicted_30d": predicted_30d,
                        "decay_rate": round(decay_rate, 2),
                        "days_to_zero": days_to_zero,
                        "action": action,
                    })

        except Exception as e:
            log.error(f"Decay prediction error for {site_slug}: {e}")

        return sorted(predictions, key=lambda x: x["decay_rate"], reverse=True)[:20]

    def forecast_revenue(self, site_slug: str) -> Dict:
        """Simple revenue forecast based on trailing click trends."""
        # This would use historical revenue data from economics engine
        # For now, return a placeholder structure
        return {
            "site": site_slug,
            "forecast_period": "next_30_days",
            "note": "Revenue forecasting requires historical data from Economics Engine",
        }

    def get_anomalies(self, limit: int = 20) -> List[Dict]:
        return self.codex.get_anomalies(limit)

    def get_decay(self, site_slug: str = None) -> List[Dict]:
        return self.codex.get_decay_predictions(site_slug)

    def get_accuracy(self) -> Dict:
        return self.codex.get_forecast_accuracy()

    def get_stats(self) -> Dict:
        return self.codex.stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predictive Intelligence Layer")
    parser.add_argument("--anomalies", action="store_true", help="Check for algorithm updates")
    parser.add_argument("--decay", help="Predict decay for a site")
    parser.add_argument("--forecast", help="Revenue forecast for a site")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    predictor = PredictiveLayer()

    if args.anomalies:
        result = predictor.detect_algorithm_update()
        if not result:
            result = {"status": "no_anomalies", "message": "No algorithm updates detected"}
    elif args.decay:
        result = predictor.predict_decay(args.decay)
    elif args.forecast:
        result = predictor.forecast_revenue(args.forecast)
    else:
        result = predictor.get_stats()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
