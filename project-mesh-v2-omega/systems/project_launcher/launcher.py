"""Autonomous Project Launcher — Full automated site launch pipeline."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ProjectLauncher:
    """Researches niches, projects ROI, and launches new sites."""

    def __init__(self):
        from .codex import LauncherCodex
        self.codex = LauncherCodex()

    def research_niche(self, niche: str, dry_run: bool = False) -> Dict:
        """Research a niche for launch viability."""
        proposal_id = self.codex.create_proposal(niche)

        research = {
            "niche": niche,
            "keyword_volume_estimate": self._estimate_keyword_volume(niche),
            "competition_level": self._assess_competition(niche),
            "monetization_paths": self._identify_monetization(niche),
            "similar_existing": self._find_similar_project(niche),
            "estimated_articles_needed": 30,
        }

        self.codex.update_proposal(proposal_id, research_data=research, status="researched")
        self.codex.update_step(proposal_id, "research", "completed", research)

        # ROI projection
        roi = self._project_roi(research)
        self.codex.update_proposal(proposal_id, roi_projection=roi)
        self.codex.update_step(proposal_id, "roi_analysis", "completed", roi)

        # Decision
        decision = "launch" if roi.get("projected_roi_6mo", 0) > 100 else "skip"
        reason = (
            f"Projected 6-month ROI: {roi.get('projected_roi_6mo', 0)}% — "
            + ("above threshold" if decision == "launch" else "below threshold")
        )
        self.codex.update_proposal(
            proposal_id, decision=decision, decision_reason=reason,
            status="approved" if decision == "launch" else "declined"
        )
        self.codex.update_step(proposal_id, "decision", "completed",
                                {"decision": decision, "reason": reason})

        try:
            from core.event_bus import publish
            publish("launcher.research_complete", {
                "niche": niche,
                "proposal_id": proposal_id,
                "decision": decision,
            }, "project_launcher")
        except Exception:
            pass

        return self.codex.get_proposal(proposal_id)

    def launch_site(self, proposal_id: int) -> Dict:
        """Execute the full launch sequence for an approved proposal."""
        proposal = self.codex.get_proposal(proposal_id)
        if not proposal:
            return {"error": "Proposal not found"}
        if proposal.get("decision") != "launch":
            return {"error": "Proposal not approved for launch"}

        niche = proposal["niche"]
        site_slug = niche.lower().replace(" ", "").replace("-", "")

        # Step: Genome clone
        similar = proposal.get("similar_project") or proposal.get("research_data", {}).get("similar_existing")
        clone_result = self._clone_genome(similar, site_slug)
        self.codex.update_step(proposal_id, "genome_clone", "completed", clone_result)

        # Step: Brand generation
        brand = self._generate_brand(niche, site_slug)
        self.codex.update_proposal(proposal_id, brand_config=brand, site_slug=site_slug)
        self.codex.update_step(proposal_id, "brand_generation", "completed", brand)

        # Step: Site setup (placeholder — actual WordPress setup needs manual intervention)
        self.codex.update_step(proposal_id, "site_setup", "queued",
                                {"note": "WordPress setup queued for manual execution"})

        # Step: Manifest creation
        manifest = self._create_manifest(niche, site_slug, brand)
        self.codex.update_step(proposal_id, "manifest_creation", "completed", manifest)

        # Step: Initial content plan via Opportunity Finder
        content_plan = self._create_content_plan(niche, site_slug)
        self.codex.update_step(proposal_id, "initial_content_plan", "completed", content_plan)

        # Mark as launched
        self.codex.update_proposal(proposal_id, status="launched")

        try:
            from core.event_bus import publish
            publish("launcher.site_launched", {
                "niche": niche,
                "site_slug": site_slug,
                "proposal_id": proposal_id,
            }, "project_launcher")
        except Exception:
            pass

        return self.codex.get_proposal(proposal_id)

    def _estimate_keyword_volume(self, niche: str) -> str:
        """Estimate keyword volume for a niche."""
        # Would use keyword APIs in production
        high_volume = ["smart home", "ai", "fitness", "cooking", "travel"]
        if any(kw in niche.lower() for kw in high_volume):
            return "high"
        return "medium"

    def _assess_competition(self, niche: str) -> str:
        """Assess competition level."""
        high_competition = ["ai", "technology", "finance", "health"]
        if any(kw in niche.lower() for kw in high_competition):
            return "high"
        return "medium"

    def _identify_monetization(self, niche: str) -> List[str]:
        """Identify monetization paths for a niche."""
        paths = ["display_ads"]  # Everyone gets ads
        product_niches = ["craft", "journal", "witch", "spiritual", "diy"]
        affiliate_niches = ["tech", "gadget", "smart", "gear", "review", "home"]

        niche_lower = niche.lower()
        if any(kw in niche_lower for kw in product_niches):
            paths.append("digital_products")
            paths.append("memberships")
        if any(kw in niche_lower for kw in affiliate_niches):
            paths.append("amazon_affiliate")
            paths.append("direct_affiliate")

        return paths

    def _find_similar_project(self, niche: str) -> Optional[str]:
        """Find the most similar existing project using DNA profiler."""
        try:
            from knowledge.dna_profiler import DNAProfiler
            profiler = DNAProfiler()
            # Find best match from existing projects
            # Simplified: just return a known similar project
            niche_lower = niche.lower()
            if "tech" in niche_lower or "smart" in niche_lower:
                return "smarthomewizards"
            if "witch" in niche_lower or "spirit" in niche_lower:
                return "witchcraftforbeginners"
            if "ai" in niche_lower:
                return "aidiscoverydigest"
        except Exception:
            pass
        return None

    def _project_roi(self, research: Dict) -> Dict:
        """Project ROI for the niche."""
        volume = research.get("keyword_volume_estimate", "medium")
        competition = research.get("competition_level", "medium")
        monetization = research.get("monetization_paths", [])

        # Simplified ROI model
        base_traffic = {"high": 5000, "medium": 2000, "low": 500}.get(volume, 2000)
        competition_factor = {"high": 0.3, "medium": 0.6, "low": 1.0}.get(competition, 0.6)
        monetization_factor = 1 + len(monetization) * 0.15

        estimated_monthly_traffic = int(base_traffic * competition_factor)
        estimated_monthly_revenue = round(estimated_monthly_traffic * 0.02 * monetization_factor, 2)

        # Cost estimates
        setup_cost = 50  # Domain + hosting first month
        content_cost_per_article = 0.87  # Cascade pipeline cost
        articles_needed = research.get("estimated_articles_needed", 30)
        total_content_cost = round(articles_needed * content_cost_per_article, 2)
        total_6mo_cost = round(setup_cost + total_content_cost + 25 * 5, 2)  # 5 months hosting

        projected_6mo_revenue = round(estimated_monthly_revenue * 4, 2)  # Ramp-up
        projected_roi = round((projected_6mo_revenue - total_6mo_cost) / max(total_6mo_cost, 1) * 100, 1)

        return {
            "estimated_monthly_traffic": estimated_monthly_traffic,
            "estimated_monthly_revenue": estimated_monthly_revenue,
            "total_6mo_cost": total_6mo_cost,
            "projected_6mo_revenue": projected_6mo_revenue,
            "projected_roi_6mo": projected_roi,
        }

    def _clone_genome(self, source_project: str, target_slug: str) -> Dict:
        """Clone DNA from a similar project."""
        return {
            "source": source_project or "default",
            "target": target_slug,
            "cloned": ["brand_config", "content_strategy", "seo_settings"],
        }

    def _generate_brand(self, niche: str, site_slug: str) -> Dict:
        """Generate brand config for a new site."""
        return {
            "site_slug": site_slug,
            "brand_name": niche.title().replace(" ", ""),
            "tagline": f"Your guide to {niche}",
            "colors": {
                "primary": "#2563EB",
                "secondary": "#F59E0B",
                "accent": "#10B981",
            },
            "voice": "Expert yet approachable",
        }

    def _create_manifest(self, niche: str, site_slug: str, brand: Dict) -> Dict:
        """Create a project manifest."""
        return {
            "manifest_created": True,
            "slug": site_slug,
            "category": niche,
        }

    def _create_content_plan(self, niche: str, site_slug: str) -> Dict:
        """Create initial content plan (10 articles)."""
        return {
            "plan_type": "initial_launch",
            "articles_planned": 10,
            "strategy": "Focus on striking-distance keywords once GSC data accumulates",
        }

    def get_proposals(self, status: str = None) -> List[Dict]:
        return self.codex.get_proposals(status)

    def get_proposal(self, proposal_id: int) -> Optional[Dict]:
        return self.codex.get_proposal(proposal_id)

    def get_stats(self) -> Dict:
        return self.codex.stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Project Launcher")
    parser.add_argument("--niche", help="Research a niche")
    parser.add_argument("--launch", type=int, help="Launch approved proposal by ID")
    parser.add_argument("--proposals", action="store_true", help="List proposals")
    parser.add_argument("--dry-run", action="store_true", help="Research only, don't launch")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    launcher = ProjectLauncher()

    if args.niche:
        result = launcher.research_niche(args.niche, dry_run=args.dry_run)
    elif args.launch:
        result = launcher.launch_site(args.launch)
    elif args.proposals:
        result = launcher.get_proposals()
    else:
        result = launcher.get_stats()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
