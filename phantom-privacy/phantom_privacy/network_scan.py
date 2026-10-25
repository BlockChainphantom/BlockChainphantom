"""A defensive nmap wrapper for auditing *your own* network.

Purpose: let you see what services your own devices expose, so you can close
what you don't need. To keep this strictly for self-audit:

* Targets must be loopback or private (RFC1918) addresses by default.
* Scanning anything public requires ``allow_public=True`` *and* is intended only
  for hosts you own or are explicitly authorised to test.
* The scan uses nmap's plain, non-evasive defaults (a basic connect scan of
  common ports). There are no stealth/evasion options here.

This module shells out to the system ``nmap``; it does not reimplement scanning.
"""

from __future__ import annotations

import ipaddress
import shutil
import socket
import subprocess
from dataclasses import dataclass

AUTHORIZATION_NOTICE = (
    "Only scan networks and devices you own or have explicit permission to test. "
    "Scanning systems without authorisation may be illegal."
)

# A small map of common ports to what they typically expose, used to annotate
# results so the output is useful to a non-expert.
PORT_NOTES: dict[int, str] = {
    21: "FTP – often unencrypted; avoid exposing it.",
    22: "SSH – fine if key-only and patched; disable password login.",
    23: "Telnet – insecure, should not be open.",
    80: "HTTP – unencrypted web; prefer HTTPS (443).",
    139: "NetBIOS – legacy file sharing; close if unused.",
    443: "HTTPS – encrypted web service.",
    445: "SMB – file sharing; a common attack target if exposed.",
    3389: "RDP – remote desktop; never expose directly to the internet.",
    5900: "VNC – remote desktop; tunnel it, don't expose it.",
    8080: "HTTP-alt – often an admin/dev interface; check it's intended.",
}


class ScanError(RuntimeError):
    """Raised when a scan cannot be performed (bad target, nmap missing, etc.)."""


@dataclass
class OpenPort:
    port: int
    protocol: str
    service: str
    note: str


@dataclass
class ScanResult:
    target: str
    open_ports: list[OpenPort]
    raw_output: str


def nmap_available() -> bool:
    """Return True if the ``nmap`` executable is on PATH."""

    return shutil.which("nmap") is not None


def _resolve_to_ip(target: str) -> ipaddress._BaseAddress:
    """Resolve a hostname/IP string to an ip address object.

    A hostname (e.g. ``localhost``) is resolved via DNS so we can check whether
    it points at a private/loopback address before scanning.
    """

    try:
        return ipaddress.ip_address(target)
    except ValueError:
        pass  # not a literal IP – try to resolve it as a hostname

    try:
        resolved = socket.gethostbyname(target)
    except OSError as exc:
        raise ScanError(f"Could not resolve target '{target}': {exc}") from exc
    return ipaddress.ip_address(resolved)


def is_self_scope(target: str) -> bool:
    """True if the target resolves to a loopback or private (RFC1918) address.

    These are the addresses that belong to 'your own' machine or LAN. A network
    (CIDR) target is also accepted when the whole range is private/loopback.
    """

    # Accept a CIDR network such as 192.168.1.0/24.
    try:
        net = ipaddress.ip_network(target, strict=False)
        return net.is_private or net.is_loopback
    except ValueError:
        pass

    addr = _resolve_to_ip(target)
    return addr.is_private or addr.is_loopback


def build_command(target: str, ports: str | None = None) -> list[str]:
    """Build a plain nmap command (TCP connect scan, no evasion)."""

    cmd = ["nmap", "-sT"]  # -sT: TCP connect scan, the most basic kind
    if ports:
        cmd += ["-p", ports]
    else:
        cmd += ["--top-ports", "100"]  # a sensible common-ports default
    cmd.append(target)
    return cmd


def _parse_open_ports(output: str) -> list[OpenPort]:
    """Parse nmap's normal output into a list of open ports.

    nmap prints lines like ``22/tcp   open   ssh``; we pick those out.
    """

    ports: list[OpenPort] = []
    for line in output.splitlines():
        line = line.strip()
        if "/tcp" not in line and "/udp" not in line:
            continue
        parts = line.split()
        if len(parts) < 3 or parts[1] != "open":
            continue
        port_proto = parts[0]  # e.g. "22/tcp"
        try:
            port_str, protocol = port_proto.split("/")
            port = int(port_str)
        except ValueError:
            continue
        service = parts[2]
        ports.append(
            OpenPort(
                port=port,
                protocol=protocol,
                service=service,
                note=PORT_NOTES.get(port, ""),
            )
        )
    return ports


def scan(
    target: str,
    ports: str | None = None,
    allow_public: bool = False,
    timeout: int = 300,
) -> ScanResult:
    """Run a defensive nmap scan against ``target`` and return parsed results.

    Raises ``ScanError`` if nmap is missing, the target is out of scope, or the
    scan fails. ``allow_public=True`` is required to scan a non-private target
    and should only be used for hosts you are authorised to test.
    """

    if not nmap_available():
        raise ScanError(
            "nmap is not installed. Install it (e.g. 'sudo apt install nmap' or "
            "'brew install nmap') and try again."
        )

    if not allow_public and not is_self_scope(target):
        raise ScanError(
            f"Target '{target}' is not a loopback/private address. "
            "This tool only scans your own network by default. "
            "Re-run with allow_public=True only for hosts you own or are "
            "authorised to test."
        )

    cmd = build_command(target, ports)
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ScanError(f"Scan timed out after {timeout}s.") from exc

    if completed.returncode != 0:
        raise ScanError(
            f"nmap exited with code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )

    output = completed.stdout
    return ScanResult(
        target=target,
        open_ports=_parse_open_ports(output),
        raw_output=output,
    )


def render(result: ScanResult) -> str:
    """Format a scan result as readable text."""

    lines = [f"Scan of {result.target}:"]
    if not result.open_ports:
        lines.append("  No open ports found in the scanned range. Good.")
        return "\n".join(lines) + "\n"

    for p in result.open_ports:
        note = f"  <- {p.note}" if p.note else ""
        lines.append(f"  {p.port}/{p.protocol}  {p.service}{note}")
    lines.append("")
    lines.append("Review each open port: close anything you don't actively need.")
    return "\n".join(lines) + "\n"
