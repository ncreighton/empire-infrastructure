"""Self-Healing Infrastructure — Master orchestrator for all healing subsystems."""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SelfHealer:
    """Master healer that coordinates service recovery, WordPress health, and traffic investigation."""

    def __init__(self):
        from .codex import HealingCodex
        from .service_healer import ServiceHealer
        from .wordpress_healer import WordPressHealer
        from .traffic_investigator import TrafficInvestigator
        from .api_key_manager import ApiKeyManager

        self.codex = HealingCodex()
        self.service_healer = ServiceHealer()
        self.wordpress_healer = WordPressHealer()
        self.traffic_investigator = TrafficInvestigator()
        self.api_key_manager = ApiKeyManager()

    def run_full_check(self) -> Dict:
        """Run all healing checks and auto-remediate where possible."""
        results = {
            "services": self._check_and_heal_services(),
            "wordpress": self._check_wordpress(),
            "api_keys": self.api_key_manager.get_summary(),
            "actions_taken": [],
        }

        # Publish event
        try:
            from core.event_bus import publish
            publish("healing.check_complete", {
                "services_healed": len(results["actions_taken"]),
                "wp_sites_checked": results["wordpress"].get("total", 0),
            }, "self_healing")
        except Exception:
            pass

        return results

    def _check_and_heal_services(self) -> Dict:
        """Check services and auto-restart any that are down."""
        try:
            from core.service_monitor import ServiceMonitor
            monitor = ServiceMonitor()
            status = monitor.check_all()
        except ImportError:
            return {"error": "ServiceMonitor not available"}

        # Load service config to check auto_heal flag
        svc_config = monitor.services

        healed = []
        skipped = []
        for svc_id, svc_status in status.items():
            # Log health check
            self.codex.log_health_check(
                svc_id,
                svc_status.get("status", "unknown"),
                svc_status.get("response_time_ms", 0),
                svc_status.get("error"),
            )

            if svc_status.get("status") == "down":
                # Skip services with auto_heal disabled
                cfg = svc_config.get(svc_id, {})
                if cfg.get("auto_heal") is False:
                    log.info(f"Service {svc_id} is DOWN but auto_heal=false — skipping")
                    skipped.append(svc_id)
                    continue

                log.warning(f"Service {svc_id} is DOWN — attempting restart")
                restart_result = self.service_healer.restart_service(svc_id)
                healed.append(restart_result)

                self.codex.log_healing_event(
                    "service_restart",
                    svc_id,
                    restart_result.get("method", "none"),
                    restart_result.get("result", "unknown"),
                    restart_result,
                )
            elif svc_status.get("status") == "healthy":
                # Reset retry count for healthy services
                self.service_healer.reset_retries(svc_id)

        healthy = sum(1 for s in status.values() if s.get("status") == "healthy")
        return {
            "total": len(status),
            "healthy": healthy,
            "down": len(status) - healthy,
            "healed": healed,
            "skipped_auto_heal": skipped,
        }

    def _check_wordpress(self) -> Dict:
        """Check all WordPress sites."""
        summary = self.wordpress_healer.get_summary()

        # Log any issues
        for site in summary.get("sites", []):
            if site.get("status") != "healthy":
                self.codex.log_healing_event(
                    "wordpress_issue",
                    site.get("site", "unknown"),
                    "diagnosed",
                    site.get("status", "unknown"),
                    {"diagnosis": site.get("diagnosis")},
                )

        return summary

    def investigate_traffic(self, site_slug: str = None) -> Dict:
        """Investigate traffic drops."""
        if site_slug:
            result = self.traffic_investigator.investigate_site(site_slug)
        else:
            result = {"investigations": self.traffic_investigator.investigate_all()}

        # Log critical findings
        if isinstance(result, dict):
            investigations = result.get("investigations", [result])
            for inv in investigations:
                if inv.get("severity") in ("critical", "warning"):
                    self.codex.log_healing_event(
                        "traffic_investigation",
                        inv.get("site", "unknown"),
                        "investigated",
                        inv.get("severity", "normal"),
                        {"findings": inv.get("findings", [])},
                    )

        return result

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent healing events."""
        return self.codex.get_recent_events(limit)

    def get_stats(self) -> Dict:
        """Overall healing system statistics."""
        return self.codex.stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Self-Healing Infrastructure")
    parser.add_argument("--check", action="store_true", help="Run full healing check")
    parser.add_argument("--services", action="store_true", help="Check services only")
    parser.add_argument("--wordpress", action="store_true", help="Check WordPress sites only")
    parser.add_argument("--traffic", help="Investigate traffic for a site (or 'all')")
    parser.add_argument("--history", action="store_true", help="Show healing history")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    import json
    logging.basicConfig(level=logging.INFO)
    healer = SelfHealer()

    if args.check or (not any([args.services, args.wordpress, args.traffic, args.history])):
        result = healer.run_full_check()
    elif args.services:
        result = healer._check_and_heal_services()
    elif args.wordpress:
        result = healer._check_wordpress()
    elif args.traffic:
        site = None if args.traffic == "all" else args.traffic
        result = healer.investigate_traffic(site)
    elif args.history:
        result = healer.get_history()
    else:
        result = healer.run_full_check()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
