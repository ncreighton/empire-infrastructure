"""
affiliate-link-manager -- Amazon Product Advertising API wrapper.
Extracted from config/amazon_paapi.json patterns.

Provides:
- AmazonPAAPI: Product search, lookup, and affiliate link generation
- format_affiliate_link(): build tagged Amazon URLs
- inject_affiliate_links(): find and tag Amazon links in HTML content

Usage:
    api = AmazonPAAPI.from_config("/path/to/amazon_paapi.json")
    results = api.search_items("smart home hub", site_id="smarthomewizards")
    link = format_affiliate_link("B0ABCDEF12", tag="smarthomewizards-20")
"""

import os
import json
import hmac
import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

log = logging.getLogger(__name__)

PAAPI_HOST = "webservices.amazon.com"
PAAPI_REGION = "us-east-1"
PAAPI_SERVICE = "ProductAdvertisingAPI"


class AmazonPAAPI:
    """Amazon Product Advertising API v5 client.

    Requires access_key and secret_key from the Amazon Associates program.
    Partner tags are mapped per site for multi-site affiliate tracking.
    """

    def __init__(self, access_key: str, secret_key: str,
                 partner_tags: Optional[Dict[str, str]] = None,
                 marketplace: str = "www.amazon.com",
                 region: str = "us-east-1"):
        self.access_key = access_key
        self.secret_key = secret_key
        self.partner_tags = partner_tags or {}
        self.marketplace = marketplace
        self.region = region

    @classmethod
    def from_config(cls, config_path: str) -> "AmazonPAAPI":
        """Load configuration from a JSON file.

        Expected format:
        {
            "access_key": "...",
            "secret_key": "...",
            "partner_tags": {"site_id": "tag-20"},
            "marketplace": "www.amazon.com",
            "region": "us-east-1"
        }
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        data = json.loads(path.read_text("utf-8"))
        return cls(
            access_key=data.get("access_key", ""),
            secret_key=data.get("secret_key", ""),
            partner_tags=data.get("partner_tags", {}),
            marketplace=data.get("marketplace", "www.amazon.com"),
            region=data.get("region", "us-east-1"),
        )

    @classmethod
    def from_env(cls) -> "AmazonPAAPI":
        """Load configuration from environment variables."""
        return cls(
            access_key=os.environ.get("AMAZON_ACCESS_KEY", ""),
            secret_key=os.environ.get("AMAZON_SECRET_KEY", ""),
            marketplace=os.environ.get("AMAZON_MARKETPLACE",
                                       "www.amazon.com"),
            region=os.environ.get("AMAZON_REGION", "us-east-1"),
        )

    def get_tag(self, site_id: str) -> str:
        """Get the partner tag for a site."""
        return self.partner_tags.get(site_id, "")

    def search_items(self, keywords: str, site_id: str = "",
                     search_index: str = "All",
                     item_count: int = 5) -> List[Dict]:
        """Search for products on Amazon.

        Args:
            keywords: Search terms
            site_id: Site ID for partner tag lookup
            search_index: Amazon department (All, Electronics, Books, etc.)
            item_count: Number of results (max 10)

        Returns:
            List of product dicts with title, asin, url, price, image.
        """
        import requests

        tag = self.get_tag(site_id) if site_id else ""

        payload = {
            "Keywords": keywords,
            "SearchIndex": search_index,
            "ItemCount": min(item_count, 10),
            "Resources": [
                "ItemInfo.Title",
                "Offers.Listings.Price",
                "Images.Primary.Large",
            ],
            "PartnerTag": tag,
            "PartnerType": "Associates",
            "Marketplace": self.marketplace,
        }

        try:
            resp = self._signed_request("SearchItems", payload)
            items = resp.get("SearchResult", {}).get("Items", [])
            return [self._parse_item(item, tag) for item in items]
        except Exception as e:
            log.error("PAAPI search failed: %s", e)
            return []

    def get_items(self, asins: List[str],
                  site_id: str = "") -> List[Dict]:
        """Look up specific products by ASIN.

        Args:
            asins: List of Amazon ASINs (max 10)
            site_id: Site ID for partner tag lookup

        Returns:
            List of product dicts.
        """
        import requests

        tag = self.get_tag(site_id) if site_id else ""

        payload = {
            "ItemIds": asins[:10],
            "Resources": [
                "ItemInfo.Title",
                "Offers.Listings.Price",
                "Images.Primary.Large",
            ],
            "PartnerTag": tag,
            "PartnerType": "Associates",
            "Marketplace": self.marketplace,
        }

        try:
            resp = self._signed_request("GetItems", payload)
            items = resp.get("ItemsResult", {}).get("Items", [])
            return [self._parse_item(item, tag) for item in items]
        except Exception as e:
            log.error("PAAPI GetItems failed: %s", e)
            return []

    def _parse_item(self, item: Dict, tag: str) -> Dict:
        """Parse a PAAPI item response into a clean dict."""
        asin = item.get("ASIN", "")
        title = (item.get("ItemInfo", {}).get("Title", {})
                 .get("DisplayValue", ""))
        url = item.get("DetailPageURL", "")
        price_info = (item.get("Offers", {}).get("Listings", [{}])[0]
                      .get("Price", {}))
        image = (item.get("Images", {}).get("Primary", {})
                 .get("Large", {}).get("URL", ""))

        return {
            "asin": asin,
            "title": title,
            "url": url,
            "price": price_info.get("DisplayAmount", ""),
            "price_value": price_info.get("Amount", 0),
            "image": image,
            "tag": tag,
        }

    def _signed_request(self, operation: str,
                        payload: Dict) -> Dict:
        """Make a signed PAAPI v5 request."""
        import requests

        url = f"https://{PAAPI_HOST}/paapi5/{operation.lower()}"
        body = json.dumps(payload)

        now = datetime.now(timezone.utc)
        date_stamp = now.strftime("%Y%m%d")
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Host": PAAPI_HOST,
            "X-Amz-Date": amz_date,
            "X-Amz-Target": f"com.amazon.paapi5.v1.{operation}",
        }

        # AWS Signature V4 (simplified)
        credential_scope = (f"{date_stamp}/{self.region}/"
                            f"{PAAPI_SERVICE}/aws4_request")

        signed_headers = ";".join(sorted(
            k.lower() for k in headers
        ))
        payload_hash = hashlib.sha256(body.encode()).hexdigest()

        canonical_request = (
            f"POST\n/paapi5/{operation.lower()}\n\n"
            + "\n".join(
                f"{k.lower()}:{v}" for k, v in sorted(headers.items())
            )
            + f"\n\n{signed_headers}\n{payload_hash}"
        )
        string_to_sign = (
            f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n"
            + hashlib.sha256(canonical_request.encode()).hexdigest()
        )

        def _sign(key, msg):
            return hmac.new(key, msg.encode(), hashlib.sha256).digest()

        k_date = _sign(
            f"AWS4{self.secret_key}".encode(), date_stamp
        )
        k_region = _sign(k_date, self.region)
        k_service = _sign(k_region, PAAPI_SERVICE)
        k_signing = _sign(k_service, "aws4_request")
        signature = hmac.new(
            k_signing, string_to_sign.encode(), hashlib.sha256
        ).hexdigest()

        headers["Authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self.access_key}/"
            f"{credential_scope}, SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )

        resp = requests.post(url, headers=headers, data=body, timeout=15)
        resp.raise_for_status()
        return resp.json()


def format_affiliate_link(asin: str, tag: str,
                          marketplace: str = "www.amazon.com") -> str:
    """Build an Amazon affiliate link from an ASIN and tag.

    Returns:
        Formatted affiliate URL like:
        https://www.amazon.com/dp/B0ABCDEF12?tag=mytag-20
    """
    return f"https://{marketplace}/dp/{asin}?tag={tag}"


def inject_affiliate_tags(html: str, tag: str) -> str:
    """Find Amazon product links in HTML and add/replace the affiliate tag.

    Handles both amazon.com/dp/ASIN and amazon.com/gp/product/ASIN patterns.
    Returns modified HTML with tagged links.
    """
    def _replace_link(match):
        url = match.group(0)
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        params["tag"] = [tag]
        new_query = urlencode(params, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    pattern = r'https?://(?:www\.)?amazon\.com/(?:dp|gp/product)/[A-Z0-9]{10}[^\s"\'<>]*'
    return re.sub(pattern, _replace_link, html)


def list_partner_tags(config_path: str) -> Dict[str, str]:
    """Load and return all partner tags from config."""
    path = Path(config_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text("utf-8"))
    return data.get("partner_tags", {})
