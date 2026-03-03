#!/usr/bin/env python3
"""Test S3 write against both local s3proxy and cloud ceph profiles."""

import boto3
from botocore.config import Config

BUCKET = "braingeneers"
KEY_PREFIX = "personal/rcurrie/test_write.txt"
BODY = b"hello from frac-bio-embed"

PROFILES = ["default", "braingeneers"]

for profile in PROFILES:
    print(f"\n--- Profile: {profile} ---")
    session = boto3.Session(profile_name=profile)

    # Try various config combinations
    configs = [
        ("bare (no config)", {}),
        ("when_required", {"config": Config(request_checksum_calculation="when_required")}),
        ("when_required + s3v4", {
            "config": Config(
                request_checksum_calculation="when_required",
                signature_version="s3v4",
            ),
        }),
    ]

    for label, kwargs in configs:
        try:
            s3 = session.client("s3", **kwargs)
            s3.put_object(Bucket=BUCKET, Key=KEY_PREFIX, Body=BODY)
            # Verify by reading back
            resp = s3.get_object(Bucket=BUCKET, Key=KEY_PREFIX)
            data = resp["Body"].read()
            assert data == BODY, f"Read back mismatch: {data!r}"
            # Clean up
            s3.delete_object(Bucket=BUCKET, Key=KEY_PREFIX)
            print(f"  {label}: OK")
        except Exception as e:
            print(f"  {label}: FAILED - {e}")
