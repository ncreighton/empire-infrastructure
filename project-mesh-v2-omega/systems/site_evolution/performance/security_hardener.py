"""
Security Hardener — Audit response headers, generate security snippets.
Covers OWASP security headers, XML-RPC, version hiding, login hardening.
"""

import logging
from typing import Dict, List

from systems.site_evolution.utils import load_site_config, get_site_domain

log = logging.getLogger(__name__)


class SecurityHardener:
    """Audit and harden WordPress security via snippets."""

    REQUIRED_HEADERS = {
        "x-frame-options": {"expected": "SAMEORIGIN", "weight": 15},
        "x-content-type-options": {"expected": "nosniff", "weight": 15},
        "x-xss-protection": {"expected": "1; mode=block", "weight": 10},
        "referrer-policy": {"expected": "strict-origin-when-cross-origin", "weight": 10},
        "permissions-policy": {"expected": None, "weight": 10},
        "strict-transport-security": {"expected": None, "weight": 15},
    }

    def audit_security(self, site_slug: str) -> Dict:
        """Check security headers and common vulnerabilities.

        Returns: {score, issues, headers_present, headers_missing}
        """
        config = load_site_config(site_slug)
        domain = get_site_domain(site_slug)
        score = 20  # Base: HTTPS active
        issues = []
        headers_present = []
        headers_missing = []

        if not domain:
            return {"score": 0, "issues": [{"type": "critical", "msg": "No domain configured"}]}

        try:
            import requests
            resp = requests.head(f"https://{domain}", timeout=10,
                                 headers={"User-Agent": "EvoSecAuditor/1.0"},
                                 allow_redirects=True)

            # Check each security header
            for header, info in self.REQUIRED_HEADERS.items():
                value = resp.headers.get(header, "")
                if value:
                    headers_present.append(header)
                    score += info["weight"]
                else:
                    headers_missing.append(header)
                    severity = "warning" if info["weight"] >= 15 else "info"
                    issues.append({
                        "type": severity,
                        "msg": f"Missing security header: {header}"
                    })

            # Check for server version exposure
            server = resp.headers.get("server", "")
            if "php" in server.lower() or "apache" in server.lower():
                issues.append({"type": "info", "msg": f"Server header exposes software: {server}"})

            # Check for WP version in HTML
            html_resp = requests.get(f"https://{domain}", timeout=10,
                                     headers={"User-Agent": "EvoSecAuditor/1.0"})
            html = html_resp.text[:5000]

            if 'generator" content="WordPress' in html:
                issues.append({"type": "warning", "msg": "WordPress version exposed in meta generator tag"})
            else:
                score += 5

            # Check XML-RPC
            try:
                xmlrpc_resp = requests.head(f"https://{domain}/xmlrpc.php", timeout=5)
                if xmlrpc_resp.status_code == 200:
                    issues.append({"type": "warning", "msg": "XML-RPC endpoint accessible"})
                else:
                    score += 5
            except requests.RequestException:
                score += 5  # Not accessible = good

        except Exception as e:
            issues.append({"type": "critical", "msg": f"Security audit request failed: {e}"})

        return {
            "site_slug": site_slug,
            "score": min(100, max(0, score)),
            "issues": issues,
            "headers_present": headers_present,
            "headers_missing": headers_missing,
        }

    def generate_security_headers_snippet(self, site_slug: str) -> str:
        """PHP snippet to add security headers."""
        return f"""<?php
/**
 * Security Headers — {site_slug}
 * Adds critical OWASP security headers to every response.
 */
function evo_security_headers() {{
    if (headers_sent()) return;

    header('X-Frame-Options: SAMEORIGIN');
    header('X-Content-Type-Options: nosniff');
    header('X-XSS-Protection: 1; mode=block');
    header('Referrer-Policy: strict-origin-when-cross-origin');
    header('Permissions-Policy: camera=(), microphone=(), geolocation=()');

    // HSTS (only if already on HTTPS)
    if (is_ssl()) {{
        header('Strict-Transport-Security: max-age=31536000; includeSubDomains');
    }}
}}
add_action('send_headers', 'evo_security_headers');
"""

    def generate_disable_xmlrpc_snippet(self) -> str:
        """PHP snippet to disable XML-RPC (prevents brute force attacks)."""
        return """<?php
/**
 * Disable XML-RPC — Prevents brute force attacks via xmlrpc.php.
 */
add_filter('xmlrpc_enabled', '__return_false');

// Also remove the XML-RPC endpoint from discovery
remove_action('wp_head', 'rsd_link');
remove_action('xmlrpc_rsd_apis', 'rest_output_rsd');

// Block direct access to xmlrpc.php
function evo_block_xmlrpc() {
    if (defined('XMLRPC_REQUEST') && XMLRPC_REQUEST) {
        wp_die('XML-RPC is disabled', 'Forbidden', array('response' => 403));
    }
}
add_action('init', 'evo_block_xmlrpc');
"""

    def generate_hide_wp_version_snippet(self) -> str:
        """PHP snippet to remove WordPress version from head and feeds."""
        return """<?php
/**
 * Hide WordPress Version — Remove version numbers from all outputs.
 */
// Remove generator meta tag
remove_action('wp_head', 'wp_generator');

// Remove version from RSS feeds
function evo_remove_version_from_feeds() {
    return '';
}
add_filter('the_generator', 'evo_remove_version_from_feeds');

// Remove version from scripts and styles
function evo_remove_version_from_assets($src) {
    if (strpos($src, 'ver=') !== false) {
        $src = remove_query_arg('ver', $src);
    }
    return $src;
}
add_filter('style_loader_src', 'evo_remove_version_from_assets', 9999);
add_filter('script_loader_src', 'evo_remove_version_from_assets', 9999);
"""

    def generate_login_hardening_snippet(self) -> str:
        """PHP snippet for generic login errors + simple rate limiting."""
        return """<?php
/**
 * Login Hardening — Generic error messages and basic rate limiting.
 */

// Generic login error (don't reveal if username exists)
function evo_generic_login_error($error) {
    return '<strong>Error:</strong> Invalid credentials. Please try again.';
}
add_filter('login_errors', 'evo_generic_login_error');

// Simple login rate limiting via transients
function evo_login_rate_limit($user, $password) {
    $ip = $_SERVER['REMOTE_ADDR'];
    $key = 'evo_login_attempts_' . md5($ip);
    $attempts = (int) get_transient($key);

    if ($attempts >= 5) {
        return new WP_Error(
            'too_many_attempts',
            'Too many login attempts. Please wait 15 minutes.'
        );
    }

    return $user;
}
add_filter('authenticate', 'evo_login_rate_limit', 30, 2);

// Increment counter on failed login
function evo_login_failed($username) {
    $ip = $_SERVER['REMOTE_ADDR'];
    $key = 'evo_login_attempts_' . md5($ip);
    $attempts = (int) get_transient($key);
    set_transient($key, $attempts + 1, 15 * MINUTE_IN_SECONDS);
}
add_action('wp_login_failed', 'evo_login_failed');

// Reset counter on successful login
function evo_login_success($user_login, $user) {
    $ip = $_SERVER['REMOTE_ADDR'];
    delete_transient('evo_login_attempts_' . md5($ip));
}
add_action('wp_login', 'evo_login_success', 10, 2);
"""

    def generate_all_security_snippets(self, site_slug: str) -> Dict[str, str]:
        """Generate all 4 security snippets as a dict."""
        return {
            "security_headers": self.generate_security_headers_snippet(site_slug),
            "disable_xmlrpc": self.generate_disable_xmlrpc_snippet(),
            "hide_wp_version": self.generate_hide_wp_version_snippet(),
            "login_hardening": self.generate_login_hardening_snippet(),
        }
