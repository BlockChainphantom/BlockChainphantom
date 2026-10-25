"""Tests for the network-scan helpers.

These tests don't run a real nmap scan – they exercise the scope checks,
command building and output parsing, which is where the safety-relevant logic
lives.
"""

import pytest

from phantom_privacy import network_scan


@pytest.mark.parametrize(
    "target",
    ["127.0.0.1", "192.168.1.1", "10.0.0.5", "172.16.0.1", "192.168.1.0/24"],
)
def test_private_and_loopback_are_in_scope(target):
    assert network_scan.is_self_scope(target) is True


@pytest.mark.parametrize("target", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
def test_public_addresses_are_out_of_scope(target):
    assert network_scan.is_self_scope(target) is False


def test_scan_rejects_public_target_without_allow_public(monkeypatch):
    # Pretend nmap exists so we reach the scope check.
    monkeypatch.setattr(network_scan, "nmap_available", lambda: True)
    with pytest.raises(network_scan.ScanError, match="not a loopback/private"):
        network_scan.scan("8.8.8.8", allow_public=False)


def test_scan_errors_clearly_when_nmap_missing(monkeypatch):
    monkeypatch.setattr(network_scan, "nmap_available", lambda: False)
    with pytest.raises(network_scan.ScanError, match="not installed"):
        network_scan.scan("127.0.0.1")


def test_build_command_defaults_to_top_ports():
    cmd = network_scan.build_command("127.0.0.1")
    assert cmd[0] == "nmap"
    assert "-sT" in cmd
    assert "--top-ports" in cmd
    assert cmd[-1] == "127.0.0.1"


def test_build_command_with_explicit_ports():
    cmd = network_scan.build_command("127.0.0.1", ports="1-1024")
    assert "-p" in cmd
    assert "1-1024" in cmd
    assert "--top-ports" not in cmd


def test_parse_open_ports_picks_out_open_lines():
    sample = """
Starting Nmap
Nmap scan report for localhost (127.0.0.1)
PORT     STATE  SERVICE
22/tcp   open   ssh
80/tcp   closed http
443/tcp  open   https
Nmap done
""".strip()
    ports = network_scan._parse_open_ports(sample)
    found = {(p.port, p.service) for p in ports}
    assert found == {(22, "ssh"), (443, "https")}
    # The known port 22 carries an explanatory note.
    ssh = next(p for p in ports if p.port == 22)
    assert ssh.note  # non-empty note from PORT_NOTES
