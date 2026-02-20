# ZimmWriter WordPress Integration Guide

> Source: https://www.rankingtactics.com/zimmwriter-wordpress-integration/

## Overview

ZimmWriter offers direct WordPress integration allowing connection of up to 999 WordPress sites for automatic blog post uploads.

## WordPress Setup Requirements

### Three Essential Fields

#### 1. WordPress Site URL

Format required: `https://yoursitename.com` without trailing paths like `/wp-admin/`. Only HTTPS is supported, as it is also an SEO ranking factor.

#### 2. WordPress Username

Requires either an admin account or one with new post creation permissions.

#### 3. WordPress App Password

This differs from the account password. Modern WordPress versions allow creating multiple "app" passwords, providing secure third-party access without full account exposure.

## App Password Creation Steps

1. Navigate to `https://www.yoursite.com/wp-admin/users.php`
2. Select and edit a user with posting privileges
3. Scroll down on their profile page to locate the **Application Passwords** section
4. Enter an application name (e.g., "ZimmWriter")
5. Click **"Add New Application Password"** button
6. Save the generated password securely -- it cannot be retrieved later
7. Enter this password into ZimmWriter

## Configuration Management

- Users can store up to **10 saved WordPress configurations**
- Options to save, update, or delete entries
- Existing configurations can be modified by selecting from a dropdown menu

## Common Issues & Solutions

### Incomplete Uploads

Some Windows computers or webservers randomly block connections. This likely indicates firewall issues requiring server or system-level debugging.

### YouTube Embeds Not Appearing

The user account must have **"editor"** or **"administrator"** access. "Author" permissions lack iFrame embed functionality needed for YouTube video insertion.

## ZimmWriter Fields Reference

| Field | Format | Example |
|-------|--------|---------|
| WordPress Site URL | `https://yoursitename.com` | `https://smarthomewizards.com` |
| WordPress Username | Admin or posting user | `admin` |
| WordPress App Password | Generated in wp-admin | `abcd 1234 efgh 5678 ijkl 9012` |

## Notes

- HTTPS is required (no HTTP support)
- App passwords use space-separated groups of 4 characters
- App passwords can be revoked at any time from the WordPress user profile
- Up to 999 WordPress sites can be configured
- Up to 10 saved configurations can be stored in ZimmWriter
